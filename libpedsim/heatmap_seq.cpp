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

// TODO: Free this
int** heatmap;
int** scaled_heatmap;

void Heatmap::setupHeatmapSeq() {
  int* hm = new int[(LENGTH - cpu_start) * LENGTH];
  heatmap = new int*[LENGTH];
  for (int i = cpu_start; i < LENGTH; i++) {
    heatmap[i] = hm + LENGTH * (i - cpu_start);
  }

  int* shm = new int[(SCALED_LENGTH - cpu_scaled_start) * SCALED_LENGTH];
  scaled_heatmap = new int*[SCALED_LENGTH];
  for (int i = cpu_scaled_start; i < SCALED_LENGTH; i++) {
    scaled_heatmap[i] = shm + SCALED_LENGTH * (i - cpu_scaled_start);
  }
}

void Heatmap::updateHeatmapSeq() {
  auto start = std::chrono::high_resolution_clock::now();

  for (int x = 0; x < LENGTH; x++) {
    for (int y = cpu_start; y < LENGTH; y++) {
      // heat fades
      heatmap[y][x] *= 0.80;
    }
  }

  int n_agents = agents_soa->getNumAgents();
  const int* desired_pos_x = agents_soa->getDesiredPosX();
  const int* desired_pos_y = agents_soa->getDesiredPosY();

  for (int i = 0; i < n_agents; i++) {
    int x = desired_pos_x[i];
    int y = desired_pos_y[i];

    if (x < 0 || x >= LENGTH || y < cpu_start || y >= LENGTH) {
      continue;
    }

    // intensify heat for better color results
    heatmap[y][x] += 40;
  }

  for (int x = 0; x < LENGTH; x++) {
    for (int y = cpu_start; y < LENGTH; y++) {
      heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
    }
  }

  for (int y = cpu_start; y < LENGTH; y++) {
    for (int x = 0; x < LENGTH; x++) {
      int value = heatmap[y][x];
      for (int cellY = 0; cellY < CELL_SIZE; cellY++) {
        for (int cellX = 0; cellX < CELL_SIZE; cellX++) {
          // int idx_y = y * CELL_SIZE + cellY;
          // int idx_x = x * CELL_SIZE + cellX;
          // if (idx_y < 0 || idx_y >= )
          // if (y)
          scaled_heatmap[y * CELL_SIZE + cellY][x * CELL_SIZE + cellX] = value;
        }
      }
    }
  }

  for (int i = cpu_scaled_start + 2 + 1; i < SCALED_LENGTH - 2;
       i++) {  // +1 to not write on top of GPU
    for (int j = 2; j < SCALED_LENGTH - 2; j++) {
      int sum = 0;
      for (int k = -2; k < 3; k++) {
        for (int l = -2; l < 3; l++) {
          sum += W[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
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
  cout << "Average time: " << total_heatmap_seq_ms / n_steps << " ms\n" << endl;
}

void Heatmap::freeHeatmapSeq() {
  delete[] heatmap[cpu_start];
  delete[] heatmap;
  delete[] scaled_heatmap[cpu_scaled_start];
  delete[] scaled_heatmap;
}