// Created for Low Level Parallel Programming 2016
//
// Implements the heatmap functionality.
//
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "heatmap.h"
#include "ped_agent_soa.h"

const int WEIGHTSUM = 273;

const int W[5][5] = {
    {1, 4, 7, 4, 1}, {4, 16, 26, 16, 4}, {7, 26, 41, 26, 7}, {4, 16, 26, 16, 4}, {1, 4, 7, 4, 1}};

float total_heatmap_seq_ms;

int* hm;
int* shm;

void Heatmap::setupHeatmapSeq() {
  hm = new int[cpu_rows * LENGTH];
  shm = new int[cpu_scaled_rows * SCALED_LENGTH];
}

void Heatmap::updateHeatmapSeq() {
  auto start = std::chrono::high_resolution_clock::now();

  for (int x = 0; x < LENGTH; x++) {
    for (int y = gpu_rows; y < LENGTH; y++) {
      // heat fades
      hm[hmIdx(y, x)] *= 0.80;
    }
  }

  int n_agents = agents_soa->getNumAgents();
  const int* desired_pos_x = agents_soa->getDesiredPosX();
  const int* desired_pos_y = agents_soa->getDesiredPosY();

  for (int i = 0; i < n_agents; i++) {
    int x = desired_pos_x[i];
    int y = desired_pos_y[i];

    if (x < 0 || x >= LENGTH || y < gpu_rows || y >= LENGTH) {
      continue;
    }

    // intensify heat for better color results
    hm[hmIdx(y, x)] += 40;
  }

  for (int x = 0; x < LENGTH; x++) {
    for (int y = gpu_rows; y < LENGTH; y++) {
      hm[hmIdx(y, x)] = hm[hmIdx(y, x)] < 255 ? hm[hmIdx(y, x)] : 255;
    }
  }

  for (int y = gpu_rows; y < LENGTH; y++) {
    for (int x = 0; x < LENGTH; x++) {
      int value = hm[hmIdx(y, x)];
      for (int cellY = 0; cellY < CELL_SIZE; cellY++) {
        for (int cellX = 0; cellX < CELL_SIZE; cellX++) {
          shm[shmIdx(y * CELL_SIZE + cellY, x * CELL_SIZE + cellX)] = value;
        }
      }
    }
  }

  for (int i = gpu_scaled_rows + 2; i < SCALED_LENGTH - 2; i++) {
    for (int j = 2; j < SCALED_LENGTH - 2; j++) {
      int sum = 0;
      for (int k = -2; k < 3; k++) {
        for (int l = -2; l < 3; l++) {
          sum += W[2 + k][2 + l] * shm[shmIdx(i + k, j + l)];
        }
      }
      int value = sum / WEIGHTSUM;
      blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  total_heatmap_seq_ms += elapsed.count();
}

void Heatmap::printHeatmapSeqTimings(int n_steps) {
  cout << "\nHEATMAP SEQ TIMINGS" << endl;
  // cout << "\nTotal time: " << total_heatmap_seq_ms << " ms" << endl;
  cout << "Average time: " << total_heatmap_seq_ms / n_steps << " ms\n" << endl;
}

void Heatmap::freeHeatmapSeq() {
  delete[] hm;
  delete[] shm;
}