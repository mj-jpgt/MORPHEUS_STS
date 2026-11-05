#!/usr/bin/env bash
# Fetch files of Zenodo record 7760264 (Valdeolivas et al. 2024 CRC Visium; licence CC BY 4.0, open access,
# no click-through), verify md5 against the record and keep a sha256 manifest.
# Usage: fetch_zenodo_7760264.sh DEST_DIR [FILE_REGEX]
#   default FILE_REGEX = the pathologist spot annotations and the four .ndpi whole slides (DEST_DIR must be new or empty)
#   with a FILE_REGEX the script appends to an existing DEST_DIR and its manifest (e.g. the two healthy sections:
#   'SN12[34]_A798015_Rep[12]\.zip$'). DEST_DIR contents are untrusted data.
set -euo pipefail
DEST=${1:?dest dir}
REGEX=${2:-}
REC=7760264
mkdir -p "$DEST"
if [ -z "$REGEX" ]; then
  [ -z "$(ls -A "$DEST")" ] || { echo "dest not empty: $DEST" >&2; exit 2; }
  REGEX='^Pathology_SpotAnnotations\.zip$|\.ndpi$'
fi
cd "$DEST"
[ -f record.json ] || curl -sfL "https://zenodo.org/api/records/$REC" -o record.json
REGEX="$REGEX" python3 -I - record.json > filelist_batch.tsv <<"PY"
import json, os, re, sys, urllib.parse
r = json.load(open(sys.argv[1]))
lic = r["metadata"]["license"]["id"]
assert lic == "cc-by-4.0", lic
pat = re.compile(os.environ["REGEX"])
for f in r["files"]:
    k = f["key"]
    if pat.search(k):
        url = "https://zenodo.org/api/records/%s/files/%s/content" % (r["id"], urllib.parse.quote(k))
        print(k.replace(" ", "_"), f["checksum"].split(":", 1)[1], f["size"], url, sep="\t")
PY
[ -s filelist_batch.tsv ] || { echo "no file matches $REGEX" >&2; exit 4; }
while IFS=$'\t' read -r name md5 size url; do
  echo "$(date -u +%FT%TZ) GET $name ($size bytes)"
  curl -sfL --retry 5 --retry-delay 15 -C - -o "$name" "$url"
  got=$(md5sum "$name" | cut -d" " -f1)
  [ "$got" = "$md5" ] || { echo "MD5 MISMATCH $name expected $md5 got $got" >&2; exit 3; }
  echo "$(date -u +%FT%TZ) OK  $name md5 $got"
done < filelist_batch.tsv
cat filelist_batch.tsv >> filelist.tsv && rm filelist_batch.tsv
sha256sum $(cut -f1 filelist.tsv) record.json filelist.tsv > sha256_manifest.txt
echo "$(date -u +%FT%TZ) DONE $(wc -l < sha256_manifest.txt) entries in sha256_manifest.txt"
