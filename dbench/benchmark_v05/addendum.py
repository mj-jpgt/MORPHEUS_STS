"""python -m dbench.benchmark_v05.addendum N "note" file1 file2 ... : hash files into runs/benchmark_v05/freeze_addendum_N.json,
chained to the previous addendum (v04 addendum 12 for N = 13)."""
import datetime, hashlib, json, pathlib, sys
H = pathlib.Path.home() / "morpheus_work_bench"
V4, V5 = H / "runs/development/benchmark_v04", H / "runs/benchmark_v05"
n, note, files = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
prev = json.loads((V4 / "freeze_addendum_12.json").read_text()) if n == 13 else json.loads((V5 / f"freeze_addendum_{n-1}.json").read_text())
doc = {"schema": "morpheus_freeze_addendum/v1", "addendum_number": n,
       "original_freeze_hash_sha256": prev["original_freeze_hash_sha256"], "previous_addendum_sha256": prev["addendum_hash_sha256"],
       "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
       "items": {f: hashlib.sha256((H / f).read_bytes()).hexdigest() for f in files}, "note": note}
doc["addendum_hash_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
V5.mkdir(parents=True, exist_ok=True)
(V5 / f"freeze_addendum_{n}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
(V5 / f"freeze_addendum_{n}.sha256").write_text(doc["addendum_hash_sha256"] + "\n")
print(n, doc["addendum_hash_sha256"])
