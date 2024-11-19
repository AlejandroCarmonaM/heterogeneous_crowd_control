#!/usr/bin/env bash

# Exit if anything returns an error
set -e

if [ "$(basename "$PWD")" != "benchmarking" ]; then
    cd benchmarking
fi

python3 benchmark_demo.py

python3 plot_demo.py
