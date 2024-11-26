#include <cstdio>
#include <iostream>

#include "heatmap.h"
#include "ped_agent_soa.h"
#include "ped_model.h"

using namespace Ped;

/*GLOBAL VARS FOR tickCUDA */
int* d_agents_x;
int* d_agents_y;

int* d_destination_x;
int* d_destination_y;
int* d_destination_r;

int* d_current_waypoint;
int* d_waypoint_start_offset;
Waypoint* d_waypoints;

int numBlocks;

/**********************************************
 * HELPER FUNCTIONS
 * *********************************************/
#define TIME_KERNELS

#ifdef TIME_KERNELS

#define CUDA_TIME(milliseconds, ...)                      \
  {                                                       \
    cudaEvent_t start, stop;                              \
    cudaEventCreate(&start);                              \
    cudaEventCreate(&stop);                               \
    cudaEventRecord(start);                               \
                                                          \
    __VA_ARGS__; /* This captures the entire operation */ \
                                                          \
    cudaEventRecord(stop);                                \
    cudaEventSynchronize(stop);                           \
    cudaEventElapsedTime(&(milliseconds), start, stop);   \
                                                          \
    cudaEventDestroy(start);                              \
    cudaEventDestroy(stop);                               \
  }

#else

#define CUDA_TIME(milliseconds, ...) __VA_ARGS__;

#endif

#define cudaCheckError(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

/************************************************************************************************
 * tickCUDA implementation *****************************************************************
 * **********************************************************************************************
 */

__global__ void tickCUDA(int* agents_x, int* agents_y, int* destination_x, int* destination_y,
                         int* destination_r, int* current_waypoint, int* waypoint_start_offset,
                         Waypoint* waypoints, int n_agents) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_agents) {
    // Compute if agent reached its current destination
    // Agents always have a set destination (they are initialized with one)
    double diffX = destination_x[i] - agents_x[i];
    double diffY = destination_y[i] - agents_y[i];
    double length = sqrt(diffX * diffX + diffY * diffY);
    bool agentReachedDestination = length < destination_r[i];

    if (agentReachedDestination) {
      int num_waypoints = waypoint_start_offset[i + 1] - waypoint_start_offset[i];
      // Only change destination if agent has more than 1 total waypoints
      if (num_waypoints > 1) {
        // Set destination to next waypoint
        current_waypoint[i] = (current_waypoint[i] + 1) % num_waypoints;
        Waypoint* w = &waypoints[waypoint_start_offset[i] + current_waypoint[i]];
        destination_x[i] = w->x;
        destination_y[i] = w->y;
        destination_r[i] = w->r;

        // Update diff and length to new destination
        diffX = destination_x[i] - agents_x[i];
        diffY = destination_y[i] - agents_y[i];
        length = sqrt(diffX * diffX + diffY * diffY);
      }
    }

    // Round double to nearest integer and move agent
    agents_x[i] = agents_x[i] + __double2int_rn(diffX / length);
    agents_y[i] = agents_y[i] + __double2int_rn(diffY / length);
  }
}
void Ped::TagentSoA::setupCUDA() {
  // Malloc on device
  cudaMalloc(&d_agents_x, n_agents * sizeof(int));
  cudaMalloc(&d_agents_y, n_agents * sizeof(int));

  cudaMalloc(&d_destination_x, n_agents * sizeof(int));
  cudaMalloc(&d_destination_y, n_agents * sizeof(int));
  cudaMalloc(&d_destination_r, n_agents * sizeof(int));

  cudaMalloc(&d_current_waypoint, n_agents * sizeof(int));
  cudaMalloc(&d_waypoint_start_offset, (n_agents + 1) * sizeof(int));
  cudaMalloc(&d_waypoints, total_waypoints * sizeof(Waypoint));

  // Move data to device
  cudaMemcpy(d_agents_x, agents_x, n_agents * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_agents_y, agents_y, n_agents * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_destination_x, destination_x, n_agents * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_destination_y, destination_y, n_agents * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_destination_r, destination_r, n_agents * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_current_waypoint, current_waypoint, n_agents * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_waypoint_start_offset, waypoint_start_offset, (n_agents + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_waypoints, waypoints, total_waypoints * sizeof(Waypoint), cudaMemcpyHostToDevice);

  numBlocks = ceil(n_agents / (float)THREADS_PER_BLOCK);
}

void Ped::TagentSoA::callTickCUDA() {
  tickCUDA<<<numBlocks, THREADS_PER_BLOCK>>>(d_agents_x, d_agents_y, d_destination_x,
                                             d_destination_y, d_destination_r, d_current_waypoint,
                                             d_waypoint_start_offset, d_waypoints, n_agents);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  cudaMemcpy(agents_x, d_agents_x, n_agents * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(agents_y, d_agents_y, n_agents * sizeof(int), cudaMemcpyDeviceToHost);
}

void Ped::TagentSoA::freeCUDA() {
  cudaFree(d_agents_x);
  cudaFree(d_agents_y);

  cudaFree(d_destination_x);
  cudaFree(d_destination_y);
  cudaFree(d_destination_r);

  cudaFree(d_current_waypoint);
  cudaFree(d_waypoint_start_offset);
  cudaFree(d_waypoints);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }
}

/************************************************
 * Heatmap CUDA implementation
 ***********************************************/

int* d_heatmap;

int* d_scaled_heatmap;

int* d_blurred_heatmap;

int* d_desired_pos_x;
int* d_desired_pos_y;

/**********************************************
SETUP AND CLEANING FUNCTONALITY FOR HEATMAP
***********************************************/

void Heatmap::setupHeatmapCUDA() {
  int n_agents = agents_soa->getNumAgents();
  cudaMalloc(&d_desired_pos_x, n_agents * sizeof(int));
  cudaMalloc(&d_desired_pos_y, n_agents * sizeof(int));

  cudaMalloc(&d_heatmap, gpu_rows * LENGTH * sizeof(int));
  cudaMalloc(&d_scaled_heatmap, gpu_scaled_rows * SCALED_LENGTH * sizeof(int));
  cudaMalloc(&d_blurred_heatmap, gpu_scaled_rows * SCALED_LENGTH * sizeof(int));
}

void Heatmap::freeHeatmapCUDA() {
  cudaFree(d_desired_pos_x);
  cudaFree(d_desired_pos_y);

  cudaFree(d_heatmap);
  cudaFree(d_scaled_heatmap);
  cudaFree(d_blurred_heatmap);
}

/**********************************************
 * KERNELS FOR HEATMAP
 * *********************************************/

__global__ void fadeHeatmap(int* heatmap, int elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elems) {
    heatmap[idx] *= 0.80;
  }
}

__global__ void updateHeatmap(const int* pos_x, const int* pos_y, int* heatmap, int n_agents,
                              int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_agents) {
    int x = pos_x[idx];
    int y = pos_y[idx];

    if (x < 0 || x >= cols || y < 0 || y >= rows) {
      return;
    }

    atomicAdd(&heatmap[y * cols + x], 40);
  }
}

__global__ void normScaleHeatmap(int* orig_heatmap, int* scaled_heatmap, int elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < elems) {
    int value = min(orig_heatmap[idx], 255);
    orig_heatmap[idx] = value;

    int x = idx % LENGTH;
    int y = idx / LENGTH;

    for (int cellY = 0; cellY < CELL_SIZE; cellY++) {
      for (int cellX = 0; cellX < CELL_SIZE; cellX++) {
        scaled_heatmap[(y * CELL_SIZE + cellY) * SCALED_LENGTH + (x * CELL_SIZE + cellX)] = value;
      }
    }
  }
}

#define WEIGHTSUM 273
#define MASK_RADIUS 2

__constant__ int w[5][5] = {
    {1, 4, 7, 4, 1}, {4, 16, 26, 16, 4}, {7, 26, 41, 26, 7}, {4, 16, 26, 16, 4}, {1, 4, 7, 4, 1}};

__global__ void gaussianBlur(int* scaled_heatmap, int* blurred_heatmap, int rows) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ int shared_mem_tile[BLOCK_LENGTH][BLOCK_LENGTH];

  // Load data to shared memory (only load data for threads that will output a value)
  if (col < SCALED_LENGTH && row < rows) {
    shared_mem_tile[threadIdx.y][threadIdx.x] = scaled_heatmap[row * SCALED_LENGTH + col];
  } else {
    shared_mem_tile[threadIdx.y][threadIdx.x] = 0;
  }

  __syncthreads();

  // Calculate the output value
  if (col < SCALED_LENGTH && row < rows) {
    int sum = 0;
    for (int i_row = -MASK_RADIUS; i_row <= MASK_RADIUS; i_row++) {
      for (int i_col = -MASK_RADIUS; i_col <= MASK_RADIUS; i_col++) {
        // Check if cell to read is within the bounds of the shared memory array (tile)
        int shared_index_x = threadIdx.x + i_col;
        int shared_index_y = threadIdx.y + i_row;
        if (shared_index_x >= 0 && shared_index_x < BLOCK_LENGTH && shared_index_y >= 0 &&
            shared_index_y < BLOCK_LENGTH) {
          // 1) Core of the tile ->> Using shared memory
          sum += shared_mem_tile[shared_index_y][shared_index_x] *
                 w[i_row + MASK_RADIUS][i_col + MASK_RADIUS];
        } else {
          // 2) Halo of the tile ->> Using global memory
          int global_index_x = col + i_col;
          int global_index_y = row + i_row;
          // Cells outside scenario are not counted (treated as if they were 0)
          if (global_index_x >= 0 && global_index_x < SCALED_LENGTH && global_index_y >= 0 &&
              global_index_y < rows) {
            sum += scaled_heatmap[(global_index_y)*SCALED_LENGTH + global_index_x] *
                   w[i_row + MASK_RADIUS][i_col + MASK_RADIUS];
          }
        }
      }
    }
    int value = sum / WEIGHTSUM;
    blurred_heatmap[row * SCALED_LENGTH + col] = 0x00FF0000 | value << 24;
  }
}

float ms_fade = 0, ms_update = 0, ms_norm_scale = 0, ms_blur = 0, ms_memcpy_desired_pos_x = 0,
      ms_memcpy_desired_pos_y = 0, ms_memcpy_blurred_heatmap = 0;

float total_ms_fade = 0, total_ms_update = 0, total_ms_norm_scale = 0, total_ms_blur = 0,
      total_ms_memcpy_desired_pos_x = 0, total_ms_memcpy_desired_pos_y = 0,
      total_ms_memcpy_blurred_heatmap = 0;

void Heatmap::copyDesiredPosToGPU() {
  int n_agents = agents_soa->getNumAgents();

  const int* h_desired_pos_x = agents_soa->getDesiredPosX();
  const int* h_desired_pos_y = agents_soa->getDesiredPosY();

  CUDA_TIME(ms_memcpy_desired_pos_x, cudaMemcpy(d_desired_pos_x, h_desired_pos_x,
                                                n_agents * sizeof(int), cudaMemcpyHostToDevice));

  CUDA_TIME(ms_memcpy_desired_pos_y, cudaMemcpy(d_desired_pos_y, h_desired_pos_y,
                                                n_agents * sizeof(int), cudaMemcpyHostToDevice));
}

void Heatmap::updateHeatmapCUDA(std::chrono::_V2::system_clock::time_point* end) {
  // KERNEL LAUNCH VARS
  int blocksPerGrid;
  int n_agents = agents_soa->getNumAgents();

  blocksPerGrid = ceil((gpu_rows * LENGTH) / (float)THREADS_PER_BLOCK);
  CUDA_TIME(ms_fade,
            fadeHeatmap<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_heatmap, gpu_rows * LENGTH));

  blocksPerGrid = ceil(n_agents / (float)THREADS_PER_BLOCK);
  CUDA_TIME(ms_update,
            updateHeatmap<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
                d_desired_pos_x, d_desired_pos_y, d_heatmap, n_agents, gpu_rows, LENGTH));

  blocksPerGrid = ceil((gpu_rows * LENGTH) / (float)THREADS_PER_BLOCK);
  CUDA_TIME(ms_norm_scale, normScaleHeatmap<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
                               d_heatmap, d_scaled_heatmap, gpu_rows * LENGTH));

  dim3 blockDim(BLOCK_LENGTH, BLOCK_LENGTH);
  dim3 gridDim(ceil(SCALED_LENGTH / (float)BLOCK_LENGTH),
               ceil(gpu_scaled_rows / (float)BLOCK_LENGTH));
  CUDA_TIME(ms_blur, gaussianBlur<<<gridDim, blockDim>>>(d_scaled_heatmap, d_blurred_heatmap,
                                                         gpu_scaled_rows));

  // Copy back blurred heatmap to host
  // -2 pixels to not write unknown data
  CUDA_TIME(ms_memcpy_blurred_heatmap,
            cudaMemcpy(bhm, d_blurred_heatmap, (gpu_scaled_rows - 2) * SCALED_LENGTH * sizeof(int),
                       cudaMemcpyDeviceToHost));
  cudaCheckError(cudaGetLastError());

  *end = std::chrono::high_resolution_clock::now();

#ifdef TIME_KERNELS
  total_ms_fade += ms_fade;
  total_ms_update += ms_update;
  total_ms_norm_scale += ms_norm_scale;
  total_ms_blur += ms_blur;
  total_ms_memcpy_desired_pos_x += ms_memcpy_desired_pos_x;
  total_ms_memcpy_desired_pos_y += ms_memcpy_desired_pos_y;
  total_ms_memcpy_blurred_heatmap += ms_memcpy_blurred_heatmap;
#endif
}

void Heatmap::printHeatmapCUDATimings(int n_steps) {
#ifdef TIME_KERNELS
  float avg_fade = total_ms_fade / n_steps;
  float avg_update = total_ms_update / n_steps;
  float avg_norm_scale = total_ms_norm_scale / n_steps;
  float avg_blur = total_ms_blur / n_steps;
  float avg_memcpy_desired_pos_x = total_ms_memcpy_desired_pos_x / n_steps;
  float avg_memcpy_desired_pos_y = total_ms_memcpy_desired_pos_y / n_steps;
  float avg_memcpy_blurred_heatmap = total_ms_memcpy_blurred_heatmap / n_steps;

  cout << "HEATMAP CUDA TIMINGS" << endl;
  cout << "Kernel_name, avg_time (ms)" << endl;
  cout << "fadeHeatmap, " << avg_fade << endl;
  cout << "updateHeatmap, " << avg_update << endl;
  cout << "normScaleHeatmap, " << avg_norm_scale << endl;
  cout << "gaussianBlur, " << avg_blur << endl;
  cout << "memcpy_desired_pos_x, " << avg_memcpy_desired_pos_x << endl;
  cout << "memcpy_desired_pos_y, " << avg_memcpy_desired_pos_y << endl;
  cout << "memcpy_blurred_heatmap, " << avg_memcpy_blurred_heatmap << endl;
  cout << "total, "
       << avg_fade + avg_update + avg_norm_scale + avg_blur + avg_memcpy_desired_pos_x +
              avg_memcpy_desired_pos_y + avg_memcpy_blurred_heatmap
       << endl;
#endif
}
