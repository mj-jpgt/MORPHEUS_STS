#!/bin/bash
# usage: run_labels.sh STUDY  (pilot, then full with 4 forked workers, one target each)
cd ~/morpheus_work_bench
export MORPHEUS_GPU_CHUNK_ELEMS=4e7
L=runs/development/benchmark_v04/logs
~/morpheus_venv/bin/python -m dbench.benchmark_v04.labels $1 --pilot > $L/pilot_$1.out 2>&1
~/morpheus_venv/bin/python -m dbench.benchmark_v04.labels $1 --workers 4 > $L/labels_$1.out 2>&1
