#!/usr/bin/env python3
"""HEST local-patch embedder: H-Optimus-0 (timm, CLS/token global pool).

Sibling of hest_fm_embed_dinov2.py / hest_fm_embed_gigapath.py in
S/morpheus_substrate_mirror/hest/, written for the breast-IDC campaign
because no hoptimus variant of that per-backbone family existed yet (only
the ccRCC/kidney_injury pass used the ad-hoc, download-fed
`hest_route_a_hoptimus_embed.py` at
S/external/hest_patches_hoptimus_20260809/). Mirrors that script's pinned
precision/model-identity contract exactly (fp32 weights, TF32 off,
deterministic cudnn, no bf16 autocast in production) and, like
hest_fm_embed_dinov2.py, consumes local official HEST patch h5 files
(keys: img, barcode[s]) already on local disk -- no download/staging.

Output: one npz shard per sample at <output-root>/embeddings/<sample_id>.npz
with keys embeddings/barcodes/indices/sample_id, same layout as the dinov2
and gigapath shards, so hest_fm_finalize.py (unmodified, cohort-agnostic)
can assemble it into the standard
S/external/hest_fm_embeddings_<date>/<cohort>/embeddings/hoptimus/ tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import h5py
import numpy as np
import timm
import torch
from PIL import Image

MODEL_ID = "bioptimus/H-optimus-0"
MODEL_REV = "b145cc1e6c6b30d3251aa8b1f844e6974188a743"
WEIGHT_SHA256 = "bf47b8467403079798661601b6ab06533be789da48d3f8b510aaefe2fbc55cd3"
WEIGHT_BYTES = 4539305018
CFG_MEAN = (0.707223, 0.578729, 0.703617)
CFG_STD = (0.211883, 0.230117, 0.177517)
IMG_SIZE = 224
FEAT_DIM = 1536
POS_SHAPE = (1, 256, 1536)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _configure_precision() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else "n/a"
        ),
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
    }


def _gpu_preflight(mem_cap_gb: float, min_free_after_mib: int) -> dict[str, Any]:
    import subprocess

    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    free_mib = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()[0]
    )
    n_apps = 0 if not apps else len([ln for ln in apps.splitlines() if ln.strip()])
    need_mib = int(mem_cap_gb * 1024)
    info = {"compute_apps": n_apps, "free_mib": free_mib, "need_mib": need_mib, "apps_raw": apps}
    if free_mib < need_mib + min_free_after_mib:
        raise SystemExit(
            f"refuse to start: free GPU mem {free_mib} MiB < need {need_mib} MiB + "
            f"reserve {min_free_after_mib} MiB ({info})"
        )
    return info


def _to_224(img: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = int(img.shape[0]), int(img.shape[1])
    meta = {"native_h": h, "native_w": w, "resampled": False}
    if h == IMG_SIZE and w == IMG_SIZE:
        return img, meta
    meta["resampled"] = True
    out = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC), dtype=np.uint8)
    return out, meta


def _normalize(batch_u8: torch.Tensor, mean, std) -> torch.Tensor:
    x = batch_u8.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def _read_patch_keys(h5: h5py.File) -> tuple[str, str | None]:
    if "img" not in h5:
        raise SystemExit(f"h5 keys missing img: {list(h5.keys())}")
    bc_key = "barcode" if "barcode" in h5 else ("barcodes" if "barcodes" in h5 else None)
    return "img", bc_key


def _local_snapshot_dir(hf_home: str | None) -> Path:
    """Resolve the already-cached H-optimus-0 snapshot; never downloads."""
    from huggingface_hub import snapshot_download

    cache_dir = (Path(hf_home) / "hub") if hf_home else None
    snap_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REV, cache_dir=cache_dir, local_files_only=True
        )
    )
    return snap_dir


def build_model(device: torch.device, hf_home: str | None) -> tuple[torch.nn.Module, dict[str, Any]]:
    snap_dir = _local_snapshot_dir(hf_home)
    cfg = json.loads((snap_dir / "config.json").read_text(encoding="utf-8"))
    pc = cfg["pretrained_cfg"]
    mean = tuple(float(x) for x in pc["mean"])
    std = tuple(float(x) for x in pc["std"])
    if mean != CFG_MEAN or std != CFG_STD:
        raise SystemExit(f"config mean/std mismatch: mean={mean} std={std}")
    if pc.get("input_size") != [3, IMG_SIZE, IMG_SIZE]:
        raise SystemExit(f"config input_size != [3,{IMG_SIZE},{IMG_SIZE}]: {pc.get('input_size')}")

    weight = snap_dir / "pytorch_model.bin"
    if weight.stat().st_size != WEIGHT_BYTES:
        raise SystemExit(f"weight size {weight.stat().st_size} != {WEIGHT_BYTES}")
    wsha = _sha256(weight)
    if wsha != WEIGHT_SHA256:
        raise SystemExit(f"weight sha256 {wsha} != pinned {WEIGHT_SHA256}")

    model = timm.create_model(
        cfg["architecture"],
        pretrained=False,
        img_size=IMG_SIZE,
        num_classes=0,
        global_pool="token",
        init_values=1e-5,
        dynamic_img_size=False,
    )
    state = torch.load(str(weight), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise SystemExit(f"load_state_dict missing={missing!r} unexpected={unexpected!r}")
    pe = model.pos_embed.detach().cpu()
    if tuple(pe.shape) != POS_SHAPE:
        raise SystemExit(f"pos_embed shape {tuple(pe.shape)} != {POS_SHAPE}")

    prec = _configure_precision()
    model = model.to(device=device, dtype=torch.float32).eval()
    meta = {
        "model_id": MODEL_ID,
        "revision": MODEL_REV,
        "snapshot_dir": str(snap_dir),
        "weight_file": weight.name,
        "weight_sha256": wsha,
        "num_features": FEAT_DIM,
        "architecture": cfg["architecture"],
        "image_mean": list(mean),
        "image_std": list(std),
        "use_bf16_autocast": False,
        "precision_flags": prec,
    }
    return model, {"mean": mean, "std": std, **meta}


def embed_batch(model: torch.nn.Module, batch_u8: torch.Tensor, device: torch.device, mean, std) -> np.ndarray:
    x = _normalize(batch_u8, mean, std).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        out = model(x)
    emb = out.float().cpu().numpy()
    if emb.ndim != 2 or emb.shape[1] != FEAT_DIM:
        raise SystemExit(f"bad embedding shape {emb.shape}")
    if not np.isfinite(emb).all():
        raise SystemExit("non-finite embedding")
    return emb


def run_smoke(model, device, mean, std) -> dict[str, Any]:
    rng = np.random.RandomState(0)
    tile_a = np.full((IMG_SIZE, IMG_SIZE, 3), 40, dtype=np.uint8)
    tile_b = (rng.rand(IMG_SIZE, IMG_SIZE, 3) * 255).astype(np.uint8)
    tile_b[20:200, 20:200] = 220
    batch = torch.from_numpy(np.stack([tile_a, tile_b], axis=0))
    emb = embed_batch(model, batch, device, mean, std)
    diff = float(np.linalg.norm(emb[0] - emb[1]))
    single = batch[:1]
    e1 = embed_batch(model, single, device, mean, std)
    e2 = embed_batch(model, single, device, mean, std)
    same = bool(np.allclose(e1[0], e2[0], rtol=0, atol=0))
    ok = emb.shape == (2, FEAT_DIM) and bool(np.isfinite(emb).all()) and diff > 1e-3 and same
    if not ok:
        raise SystemExit(f"smoke test failed shape={emb.shape} diff={diff} same_proc_repeat={same}")
    return {"ok": ok, "l2_diff_tiles": diff, "same_process_repeat_identical": same, "shape": list(emb.shape)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="comma-separated sample ids")
    ap.add_argument("--patch-dir", required=True, help="dir containing {sample}.h5 or {sample}_patches.h5")
    ap.add_argument("--patch-suffix", default=".h5", help="filename suffix after sample id, e.g. .h5 or _patches.h5")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gpu-mem-cap-gb", type=float, default=6.0)
    ap.add_argument("--min-free-after-mib", type=int, default=6144)
    ap.add_argument("--hf-home", default="")
    ap.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="cuda (production; preflight + CUDA required) or cpu (pipeline pre-check only, no preflight)",
    )
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    out_root = Path(args.output_root)
    (out_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (out_root / "receipts").mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required")
        device = torch.device("cuda")
        preflight = _gpu_preflight(args.gpu_mem_cap_gb, args.min_free_after_mib)
    else:
        device = torch.device("cpu")
        preflight = {"skipped": "device=cpu"}

    model, meta = build_model(device, args.hf_home or os.environ.get("HF_HOME"))
    mean, std = meta["mean"], meta["std"]
    smoke = run_smoke(model, device, mean, std)
    _atomic_json(
        out_root / "verification.json",
        {"smoke_test": smoke, "model_identity": meta, "gpu_preflight": preflight, "started_utc": _utc()},
    )

    sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]
    patch_dir = Path(args.patch_dir)
    receipts = []
    t_all0 = time.time()
    total_tiles = 0
    for sid in sample_ids:
        receipt_p = out_root / "receipts" / f"{sid}.json"
        shard = out_root / "embeddings" / f"{sid}.npz"
        if receipt_p.is_file() and shard.is_file():
            r = json.loads(receipt_p.read_text(encoding="utf-8"))
            if r.get("shard_sha256") == _sha256(shard):
                print(f"skip completed {sid}", flush=True)
                receipts.append(r)
                continue

        h5_path = patch_dir / f"{sid}{args.patch_suffix}"
        if not h5_path.is_file():
            raise SystemExit(f"missing patch file for {sid}: {h5_path}")
        source_sha256 = _sha256(h5_path)
        t0 = time.time()
        with h5py.File(h5_path, "r") as h5:
            img_key, bc_key = _read_patch_keys(h5)
            n = int(h5[img_key].shape[0])
            native = tuple(int(x) for x in h5[img_key].shape[1:3])
            barcodes = (
                h5[bc_key][:].astype(str).reshape(-1)
                if bc_key is not None
                else np.asarray([f"{sid}:{i}" for i in range(n)], dtype=str)
            )
            imgs_raw = np.asarray(h5[img_key][:])

        geom_counts: dict[str, int] = {}
        imgs224 = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        resampled_any = False
        for i in range(n):
            img224, g = _to_224(imgs_raw[i])
            imgs224[i] = img224
            key = f"{g['native_h']}x{g['native_w']}"
            geom_counts[key] = geom_counts.get(key, 0) + 1
            resampled_any = resampled_any or g["resampled"]

        embs = np.empty((n, FEAT_DIM), dtype=np.float32)
        bs = args.batch_size
        for i in range(0, n, bs):
            batch = torch.from_numpy(imgs224[i : i + bs])
            embs[i : i + batch.shape[0]] = embed_batch(model, batch, device, mean, std)
        total_tiles += n

        np.savez_compressed(
            shard,
            embeddings=embs,
            barcodes=barcodes,
            indices=np.arange(n, dtype=np.int64),
            sample_id=np.asarray([sid]),
        )
        receipt = {
            "sample_id": sid,
            "n_patches": int(n),
            "embedding_dim": int(embs.shape[1]),
            "shard": str(shard),
            "shard_sha256": _sha256(shard),
            "source_file_sha256": source_sha256,
            "native_dataset_hw": list(native),
            "geom_counts": geom_counts,
            "resampled_any": resampled_any,
            "elapsed_seconds": time.time() - t0,
            "completed_utc": _utc(),
        }
        _atomic_json(receipt_p, receipt)
        receipts.append(receipt)
        print(
            json.dumps({"sample_done": sid, "n_patches": n, "elapsed_s": receipt["elapsed_seconds"]}),
            flush=True,
        )

    manifest = {
        "schema": "hest_fm_hoptimus_manifest_v1",
        "created_utc": _utc(),
        "model_identity": meta,
        "gpu_preflight": preflight,
        "samples_requested": sample_ids,
        "per_sample": receipts,
        "total_tiles": total_tiles,
        "batch_size": args.batch_size,
        "elapsed_seconds_total": time.time() - t_all0,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "h5py": h5py.version.version,
        "interpreter": sys.executable,
    }
    _atomic_json(out_root / "manifest.json", manifest)
    print(json.dumps({"status": "ok", "n_samples": len(receipts), "total_tiles": total_tiles}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
