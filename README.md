# Parallel programming and high-performance computing 2024/25

## Building

```bash
make
```

## Running demo

The simulation can be run with:

```
./demo/demo [-h|--help] [--timing-mode]  [--heatmap_(seq,par,het)] [--implementation=CUDA,vector,OMP,pthreads,sequential,col_prevent_seq,col_prevent_par)] [-n(NUM_THREADS)] SCENARIO_FILE
```

E.g.:

```
./demo/demo --timing-mode --heatmap_het --implementation=col_prevent_par -n4 demo/commute_200000.xml
```

Options are:

- `--timing-mode`: Reduce output to the terminal and don't show graphic representation.
- `--heatmap_(seq,par,het)`: Selects a heatmap implementation. If this option isn't specified, heatmap is not shown. Options are:
    - `heatmap_seq`: Heatmap is computed by a single CPU thread.
    - `heatmap_par`: Heatmap is computed in the GPU.
    - `heatmap_het`: Heatmap workload is divided between CPU (single thread) and GPU.
- `--implementation=`: Selects an implementation for moving agents. If this option isn't specified, `sequential` is the default implementation. Options are:
    - `sequential`: A single CPU thread is used.
    - `OMP`: A number of threads specified by the `-n` option are used with OpenMP.
    - `pthreads`: Same as `OMP` but threads are managed with pthreads.
    - `vector`: Same as `OMP` option, but each thread uses SIMD instructions to process 4 agents at a time.
    - `CUDA`: Agent movement is processed in the GPU.
    - `col_prevent_seq`: Same as `sequential` but with collision avoidance between agents.
    - `col_prevent_par`: Divides the scenario in 4 regions, allowing parallel agent movement (with collision avoidance) calculation with multiple threads. The number of threads to use can be specified with `-n`, but the fastest value is very likely to be `-n4`.
- `-n(NUM_THREADS)`: Sets number of threads for `OMP`, `vector`, `pthreads` and `col_prevent_par`. `NUM_THREADS` must be > 0 and <= 16.

## Acknowledgment

This project includes software from the [PEDSIM](https://github.com/chgloor/pedsim) simulator.
