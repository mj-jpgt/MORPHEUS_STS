#!/usr/bin/env bash
# Fetch the two GEO CRC replication cohorts of the second-wave programme into new empty directories.
#   fetch_geo_crc.sh gse294385 DEST   -- GSE294385_RAW.tar (7.9 GB), extracted per sample under extracted/<sample>/
#   fetch_geo_crc.sh gse280318_hd DEST -- GSE280318 Visium HD (GSE280315) per-sample supplementary files only:
#                                        filtered_feature_bc_matrix.h5, tissue_positions.parquet.gz, Metadata.parquet.gz,
#                                        tissue_lowres_image.png.gz and scalefactors_json.json.gz for the five HD sections;
#                                        no tissue_image.btf / image.tif / he_image.ome.tif
# GEO terms: no licence on the records; NCBI places no restrictions on use or distribution of GEO data (NCBI policies page).
# DEST contents are untrusted data: nothing in them is executed.
set -euo pipefail
MODE=${1:?gse294385 | gse280318_hd}
DEST=${2:?dest dir}
mkdir -p "$DEST"
[ -z "$(ls -A "$DEST")" ] || { echo "dest not empty: $DEST" >&2; exit 2; }
cd "$DEST"
get() { echo "$(date -u +%FT%TZ) GET $(basename "$1")"; curl -sfL --retry 5 --retry-delay 20 -C - -o "$(basename "$1")" "$1"; }

case "$MODE" in
  gse294385)
    base=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE294nnn/GSE294385/suppl
    get $base/filelist.txt
    curl -sfL "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385&targ=all&form=text&view=full" -o GSE294385_family_text.txt
    get $base/GSE294385_RAW.tar
    expected=$(awk -F'\t' '$1=="Archive"{print $4}' filelist.txt)
    actual=$(stat -c %s GSE294385_RAW.tar)
    [ "$expected" = "$actual" ] || { echo "SIZE MISMATCH RAW.tar expected $expected got $actual" >&2; exit 3; }
    mkdir -p samples extracted
    tar -xf GSE294385_RAW.tar -C samples
    for tgz in samples/*.tar.gz; do
      name=$(basename "$tgz" .tar.gz)
      mkdir -p "extracted/$name"
      tar -xzf "$tgz" -C "extracted/$name" --no-same-owner --no-same-permissions
    done
    cat > README.md <<"EOF"
# GSE294385 (Liu et al. 2026, Cancer Cell): colorectal cancer metastasis Visium

CRC metastasis samples, molecular only, no WSI, no annotations, site/patient labels absent in GEO.

* 37 Visium sections (GSM8903657-GSM8903693), Space Ranger outputs only: filtered_feature_bc_matrix (mtx), spatial/
  (tissue_hires_image.png <= 2000 px, tissue_lowres_image.png, CytAssist image where present, tissue_positions.csv,
  scalefactors_json.json, spatial_enrichment.csv). No full-resolution microscope H&E, so no morphology, H-Optimus-0 or
  CellViT view is possible from this record.
* No pathologist spot annotations in GEO; every sample record says only "tissue: Colorectal cancer metastasis".
  Primary / liver / lung site and patient identity are not in GEO (the paper reports 11 patients).
* GEO terms: no licence stated; NCBI places no restrictions on use or distribution (NCBI policies, Molecular Data Usage).
* Layout: GSE294385_RAW.tar (verified against filelist.txt size), samples/<GSM>_<id>.tar.gz, extracted/<GSM>_<id>/<id>/...
* Downloaded by scripts/second_wave/fetch_geo_crc.sh (Agent S, second wave); sha256_manifest.txt covers every file.
EOF
    ;;
  gse280318_hd)
    curl -sfL "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280315&targ=all&form=text&view=full" -o GSE280315_family_text.txt
    for pair in GSM8594567:P1CRC GSM8594568:P2CRC GSM8594569:P5CRC GSM8594570:P3NAT GSM8594571:P5NAT; do
      gsm=${pair%%:*}; sec=${pair##*:}
      mkdir -p "$sec"; ( cd "$sec"
        for suf in filtered_feature_bc_matrix.h5 tissue_positions.parquet.gz tissue_lowres_image.png.gz scalefactors_json.json.gz; do
          get "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/$gsm/suppl/${gsm}_${sec}_${suf}"
        done
        # Metadata.parquet (deconvolution / clustering labels) is served for the three CRC sections only
        get "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/$gsm/suppl/${gsm}_${sec}_Metadata.parquet.gz" \
          || { rm -f "${gsm}_${sec}_Metadata.parquet.gz"; echo "${gsm}_${sec}_Metadata.parquet.gz not served by GEO (filelist.txt lists Metadata.parquet for P1CRC, P2CRC, P5CRC only)" >> ../NOT_SERVED.txt; } )
    done
    cat > README.md <<"EOF"
# GSE280318 / GSE280315 (Oliveira et al. 2025, Nat Genet; 10x Genomics): Visium HD colorectal cancer

Visium HD sections only (8 um bins): P1CRC, P2CRC, P5CRC (tumour) and P3NAT, P5NAT (normal adjacent), FFPE.
Files per section: filtered_feature_bc_matrix.h5, tissue_positions.parquet.gz, tissue_lowres_image.png.gz,
scalefactors_json.json.gz, and for the three CRC sections only Metadata.parquet.gz (deconvolution and unsupervised
clustering labels, not pathologist annotations; GEO serves none for P3NAT and P5NAT, see NOT_SERVED.txt).
Not downloaded: tissue_image.btf (5.8-8.1 GB full-resolution microscope H&E per section), image.tif (CytAssist),
the Xenium he_image.ome.tif files, and the Visium v2 / Flex / VDJ sub-series.

Meta-training overlap: the Visium v2 section P2CRC of this study is HEST TENX152, a member of the meta_training
study tenx_demo_bowel (dbench/protocols/sprint_partitions_v01.json), so the learned search model has seen that
section; the Visium HD sections are not in HEST. Any replication on this study must use the HD sections only and
say so.

GEO terms: no licence stated on the record; NCBI places no restrictions on use or distribution; the same 10x data
on the 10x Genomics site carry Creative Commons Attribution in the HEST catalogue.
Downloaded by scripts/second_wave/fetch_geo_crc.sh (Agent S, second wave); sha256_manifest.txt covers every file.
EOF
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
find . -type f ! -name sha256_manifest.txt -print0 | sort -z | xargs -0 sha256sum > sha256_manifest.txt
echo "$(date -u +%FT%TZ) DONE $(wc -l < sha256_manifest.txt) entries in sha256_manifest.txt"
