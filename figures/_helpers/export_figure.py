from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_plot_data(data: Any, outdir: str | Path, stem: str = "plot_data") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame):
        try:
            path = outdir / f"{stem}.parquet"
            data.to_parquet(path, index=False)
        except Exception:
            path = outdir / f"{stem}.csv"
            data.to_csv(path, index=False)
        return path
    if isinstance(data, pd.Series):
        path = outdir / f"{stem}.csv"
        data.to_frame().to_csv(path, index=True)
        return path
    path = outdir / f"{stem}.npz"
    if isinstance(data, dict):
        np.savez_compressed(path, **data)
    else:
        np.savez_compressed(path, data=np.asarray(data))
    return path


def export_figure(fig, outdir: str | Path, *, stem: str = "figure", dpi: int = 450,
                  save_pdf: bool = True, save_svg: bool = True, save_png: bool = True) -> dict[str, str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if save_pdf:
        p = outdir / f"{stem}.pdf"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        outputs["pdf"] = str(p)
    if save_svg:
        p = outdir / f"{stem}.svg"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        outputs["svg"] = str(p)
    if save_png:
        p = outdir / f"{stem}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        outputs["png"] = str(p)
    return outputs


def write_manifest(outdir: str | Path, manifest: dict[str, Any]) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def copy_plot_code(outdir: str | Path, source_file: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "plot_code.py"
    shutil.copy2(source_file, target)
    return target
