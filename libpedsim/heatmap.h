#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "ped_agent_soa.h"

#define LENGTH 1024
#define TOTAL_CELLS (LENGTH * LENGTH)

#define CELL_SIZE 5
#define SCALED_LENGTH (CELL_SIZE * LENGTH)
#define SCALED_TOTAL (SCALED_LENGTH * SCALED_LENGTH)

#define BLOCK_LENGTH 16
#define THREADS_PER_BLOCK (BLOCK_LENGTH * BLOCK_LENGTH)

class Heatmap {
 public:
  // TODO: Remove NONE impl
  enum HEATMAP_IMPL { SEQ_HM, PAR_HM, HET_HM, NONE };
  static constexpr float FRACTION_GPU = 832.0f / 1024.0f;

  Heatmap(HEATMAP_IMPL impl, Ped::TagentSoA* agents_soa) : impl(impl), agents_soa(agents_soa) {
    switch (impl) {
      case SEQ_HM:
        gpu_rows = 0;
        gpu_scaled_rows = 0;
        cpu_start = 0;
        cpu_scaled_start = 0;
        printf("Heatmap CPU rows [%d:%d] (%d rows for CPU)\n", cpu_start, LENGTH - 1,
               LENGTH - cpu_start);

        setupHeatmapSeq();
        break;
      case PAR_HM:
        gpu_rows = LENGTH;
        gpu_scaled_rows = gpu_rows * CELL_SIZE;
        cpu_start = LENGTH - 1;
        cpu_scaled_start = LENGTH * CELL_SIZE;
        printf("Heatmap GPU rows [0:%d] (%d rows for GPU)\n", gpu_rows - 1, gpu_rows);

        setupHeatmapCUDA();
        break;
      case HET_HM:
        gpu_rows = ceil(round(FRACTION_GPU * LENGTH) / BLOCK_LENGTH) * BLOCK_LENGTH;
        gpu_scaled_rows = CELL_SIZE * gpu_rows;
        cpu_start = gpu_rows - 1;  // Compute 1 extra row
        cpu_scaled_start = CELL_SIZE * cpu_start;
        printf("Heatmap GPU rows [0:%d] (%d rows for GPU)\n", gpu_rows - 1, gpu_rows);
        printf("Heatmap CPU rows [%d:%d] (%d rows for CPU)\n", cpu_start, LENGTH - 1,
               LENGTH - cpu_start);

        setupHeatmapSeq();
        setupHeatmapCUDA();
        break;
      case NONE:
        return;
        break;
    }

    allocateBlurredHeatmap();
  };

  ~Heatmap() {
    if (impl == PAR_HM || impl == HET_HM) {
      cudaFreeHost(bhm);
    } else {
      delete[] bhm;
    }

    switch (impl) {
      case SEQ_HM:
        freeHeatmapSeq();
        break;
      case PAR_HM:
        freeHeatmapCUDA();
        break;
      case HET_HM:
        freeHeatmapSeq();
        freeHeatmapCUDA();
        break;
      case NONE:
        return;
        break;
    }

    delete[] blurred_heatmap;
  };

  void updateHeatmapSeq();
  void updateHeatmapCUDA(std::chrono::_V2::system_clock::time_point* end);

  int const* const* getHeatmap() { return blurred_heatmap; }

  int getHeatmapSize() { return SCALED_LENGTH; }

  HEATMAP_IMPL getHeatmapImpl() { return impl; }

  void copyDesiredPosToGPU();

  void printHeatmapSeqTimings(int n_steps);
  void printHeatmapCUDATimings(int n_steps);

 private:
  HEATMAP_IMPL impl;
  Ped::TagentSoA* agents_soa;

  int gpu_rows, gpu_scaled_rows;
  int cpu_start, cpu_scaled_start;

  int* bhm = nullptr;
  int** blurred_heatmap = nullptr;

  void allocateBlurredHeatmap() {
    if (impl == PAR_HM || impl == HET_HM) {
      cudaHostAlloc(&bhm, SCALED_LENGTH * SCALED_LENGTH * sizeof(int), cudaHostAllocDefault);
    } else {
      bhm = new int[SCALED_LENGTH * SCALED_LENGTH];
    }

    blurred_heatmap = new int*[SCALED_LENGTH];
    for (int i = 0; i < SCALED_LENGTH; i++) {
      blurred_heatmap[i] = bhm + SCALED_LENGTH * i;
    }
  }

  void setupHeatmapSeq();
  void setupHeatmapCUDA();

  void freeHeatmapSeq();
  void freeHeatmapCUDA();
};
