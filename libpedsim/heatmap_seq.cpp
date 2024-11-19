// Created for Low Level Parallel Programming 2016
//
// Implements the heatmap functionality.
//
#include <cstdlib>
#include <iostream>

#include "ped_model.h"
using namespace std;

void Ped::Model::setupHeatmapSeq() {
  cout << "Setting up heatmap" << endl;
  int* hm = (int*)calloc(SIZE * SIZE, sizeof(int));
  int* shm = (int*)malloc(SCALED_SIZE * SCALED_SIZE * sizeof(int));
  int* bhm = (int*)malloc(SCALED_SIZE * SCALED_SIZE * sizeof(int));

  heatmap = (int**)malloc(SIZE * sizeof(int*));

  scaled_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
  blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));

  for (int i = 0; i < SIZE; i++) {
    heatmap[i] = hm + SIZE * i;
  }
  for (int i = 0; i < SCALED_SIZE; i++) {
    scaled_heatmap[i] = shm + SCALED_SIZE * i;
    blurred_heatmap[i] = bhm + SCALED_SIZE * i;
  }
}

void Ped::Model::updateHeatmapSeq() {
  auto start = std::chrono::high_resolution_clock::now();
  // cout << "Updating heatmap cpu" << endl;
  /*SEC 1: Fading*/
  /*This loop iterates over every cell in the heatmap array and reduces its heat value by
   * multiplying it by 0.80. This simulates the fading of heat over time, ensuring that old data
   * gradually diminishes.*/
  for (int x = 0; x < SIZE; x++) {
    for (int y = 0; y < SIZE; y++) {
      // heat fades
      heatmap[y][x] *= 0.80;
    }
  }

  int n_agents = agents_soa->getNumAgents();
  const int* desired_pos_x = agents_soa->getDesiredPosX();
  const int* desired_pos_y = agents_soa->getDesiredPosY();

  /*SEC 2: Update Heatmap with current agents desired positions*/
  /*This loop goes through each agent and retrieves their desired x and y positions. If the position
   * is within bounds, it increments the corresponding heatmap cell by 40 to indicate the agent's
   * intention to move to that location. */
  for (int i = 0; i < n_agents; i++) {
    int x = desired_pos_x[i];
    int y = desired_pos_y[i];

    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
      continue;
    }

    // intensify heat for better color results
    heatmap[y][x] += 40;
  }

  /*2.1. Normalizing pixel values*/
  for (int x = 0; x < SIZE; x++) {
    for (int y = 0; y < SIZE; y++) {
      heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
    }
  }

  /*SEC 3: Scaling the Heatmap for Visualization*/
  /*Functionality: This nested loop scales up the heatmap by a factor defined by CELLSIZE. Each cell
   * in the original heatmap is expanded into a CELLSIZE x CELLSIZE block in the scaled_heatmap,
   * duplicating the heat value across this block.*/
  for (int y = 0; y < SIZE; y++) {
    for (int x = 0; x < SIZE; x++) {
      int value = heatmap[y][x];
      for (int cellY = 0; cellY < CELLSIZE; cellY++) {
        for (int cellX = 0; cellX < CELLSIZE; cellX++) {
          scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
        }
      }
    }
  }

  /*SEC 4: Apply Gaussian Blur Filter*/
  /*4.1. This defines a 5x5 Gaussian kernel w used for blurring the heatmap. WEIGHTSUM is the sum of
   * all weights in the kernel, used for normalization.*/
  const int w[5][5] = {
      {1, 4, 7, 4, 1}, {4, 16, 26, 16, 4}, {7, 26, 41, 26, 7}, {4, 16, 26, 16, 4}, {1, 4, 7, 4, 1}};

#define WEIGHTSUM 273
  /*5.2. This section performs convolution between the scaled_heatmap and the Gaussian kernel w. It
   * calculates the weighted sum of the neighboring pixels for each pixel in the scaled_heatmap,
   * then normalizes it by dividing by WEIGHTSUM. The result is stored in blurred_heatmap, combining
   * the heat value with a color code.*/
  for (int i = 2; i < SCALED_SIZE - 2; i++) {
    for (int j = 2; j < SCALED_SIZE - 2; j++) {
      int sum = 0;
      for (int k = -2; k < 3; k++) {
        for (int l = -2; l < 3; l++) {
          sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
        }
      }
      int value = sum / WEIGHTSUM;
      blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "updateHeatmapSeq execution time: " << elapsed.count() << " milliseconds"
            << std::endl;
}

int Ped::Model::getHeatmapSize() const { return SCALED_SIZE; }

void Ped::Model::freeHeatmapSeq() {
  cout << "Freeing heatmap cpu" << endl;
  free(heatmap[0]);
  free(heatmap);
  free(scaled_heatmap[0]);
  free(scaled_heatmap);
  free(blurred_heatmap[0]);
  free(blurred_heatmap);
}