#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "ped_agent_soa.h"

#define LENGTH 1024
#define TOTAL_CELLS (LENGTH * LENGTH)

#define CELL_SIZE 5
#define SCALED_LENGTH (CELL_SIZE * LENGTH)
#define SCALED_TOTAL (SCALED_LENGTH * SCALED_LENGTH)

#define BLOCK_LENGTH 32
#define THREADS_PER_BLOCK (BLOCK_LENGTH * BLOCK_LENGTH)

class Heatmap {
 public:
  // TODO: Remove NONE impl
  enum HEATMAP_IMPL { SEQ_HM, PAR_HM, HET_HM, NONE };

  Heatmap(HEATMAP_IMPL impl, Ped::TagentSoA* agents_soa) : impl(impl), agents_soa(agents_soa) {
    switch (impl) {
      case SEQ_HM:
        distributeRows(0);
        setupHeatmapSeq();
        break;
      case PAR_HM:
        distributeRows(1);
        setupHeatmapCUDA();
        break;
      case HET_HM:
        distributeRows(960.0f / 1024.0f);
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

    delete[] blurred_heatmap;

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
  };

  // Set number of rows assigned to GPU and GPU, aligning number of gpu rows to thread block length
  void distributeRows(float fraction_gpu) {
    gpu_rows = ceil(round(fraction_gpu * LENGTH) / BLOCK_LENGTH) * BLOCK_LENGTH;
    gpu_scaled_rows = CELL_SIZE * gpu_rows;

    cpu_rows = LENGTH - gpu_rows;
    cpu_scaled_rows = CELL_SIZE * cpu_rows;

    std::cout << "Heatmap GPU rows: " << gpu_rows << std::endl;
    std::cout << "Heatmap CPU rows: " << cpu_rows << std::endl;
  }

  void updateHeatmapSeq();
  void updateHeatmapCUDA();

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
  int cpu_rows, cpu_scaled_rows;

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

  constexpr int hmIdx(int row, int col) {
    int res = (row - gpu_rows) * LENGTH + col;
    // if (res < 0 || res >= cpu_rows * LENGTH) {
    //   std::cout << "ERROR input " << row << " " << col << " output " << row - gpu_rows << col
    //             << " -> " << res << std::endl;
    // }
    return res;
  }

  constexpr int shmIdx(int row, int col) {
    int res = (row - gpu_scaled_rows) * SCALED_LENGTH + col;
    // if (res < 0 || res >= cpu_scaled_rows * SCALED_LENGTH) {
    //   std::cout << "SCALED ERROR input " << row << " " << col << " output " << row -
    //   gpu_scaled_rows
    //             << col << " -> " << res << std::endl;
    // }
    return res;
  }
};
