#!/bin/bash
# one running job at a time: glioma, then the lung cohorts, each through the lock
~/bin/gated_launch.sh --min-gb 70 --min-gpu-mb 8000 -- ~/morpheus_work_bench/dbench/benchmark_v04/run_labels.sh gse237183_glioma
~/bin/gated_launch.sh --min-gb 90 --min-gpu-mb 8000 -- ~/morpheus_work_bench/dbench/benchmark_v04/run_labels.sh gse307534_lung_precursor_b3
~/bin/gated_launch.sh --min-gb 70 --min-gpu-mb 8000 -- ~/morpheus_work_bench/dbench/benchmark_v04/run_labels.sh nakajima_luad_b3
