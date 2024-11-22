#include <cstdio>
#include <iostream>

#include "ped_agent_soa.h"
#include "ped_model.h"
using namespace Ped;

/*CONSTANTS FOR GAUSSIAN BLUR (Heatmap CUDA) */
#define BLOCK_SIZE_1D 16
// warp size (32) multiple which performs the best in our case
#define BLOCK_SIZE (BLOCK_SIZE_1D * BLOCK_SIZE_1D)
#define MASK_RADIUS 2
#define MASK_DIM (2 * MASK_RADIUS + 1)

#define FILTER_W 5
#define FILTER_H 5
// TODO: CHANGE THIS TO CONSTEXPR

#define BLOCKS_PER_SIZE (SIZE / BLOCK_SIZE_1D)
#define BLOCKS_PER_SCALED_SIZE (SCALED_SIZE / BLOCK_SIZE_1D)

#define ELEMS_PER_DIVISION (BLOCKS_PER_SIZE * BLOCK_SIZE)
#define ELEMS_PER_SCALED_DIVISION (BLOCKS_PER_SCALED_SIZE * BLOCK_SIZE)

#define SIZE_GPU_WITHOUT_CEIL (int)(FRACTION_GPU * SIZE * SIZE)
#define SCALED_SIZE_GPU_WITHOUT_CEIL (int)(FRACTION_GPU * SCALED_SIZED * SCALED_SIZED)

#define SIZE_GPU \
  SIZE_GPU_WITHOUT_CEIL + (ELEMS_PER_DIVISION - (SIZE_GPU_WITHOUT_CEIL % ELEMS_PER_DIVISION))
#define SCALED_SIZE_GPU          \
  SCALED_SIZE_GPU_WITHOUT_CEIL + \
      (ELEMS_PER_SCALED_DIVISION - (SCALED_SIZE_GPU_WITHOUT_CEIL % ELEMS_PER_SCALED_DIVISION))

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

/*GLOBAL VARS FOR TIMING KERNELS*/
float total_ms_fade = 0, total_ms_update = 0, total_ms_norm_scale = 0, total_ms_blur = 0,
      total_ms_memcpy_desired_pos_x = 0, total_ms_memcpy_desired_pos_y = 0,
      total_ms_memcpy_blurred_heatmap = 0, total_time_GPU = 0;

/**********************************************
 * HELPER FUNCTIONS
 * *********************************************/
#define MEASURE_KERNEL_EXECUTION(milliseconds, ...)           \
  {                                                           \
    cudaEvent_t start, stop;                                  \
    cudaEventCreate(&start);                                  \
    cudaEventCreate(&stop);                                   \
    cudaEventRecord(start);                                   \
                                                              \
    __VA_ARGS__; /* This captures the entire kernel launch */ \
                                                              \
    cudaEventRecord(stop);                                    \
    cudaEventSynchronize(stop);                               \
    cudaEventElapsedTime(&(milliseconds), start, stop);       \
                                                              \
    cudaEventDestroy(start);                                  \
    cudaEventDestroy(stop);                                   \
  }

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

  numBlocks = (n_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

void Ped::TagentSoA::callTickCUDA() {
  tickCUDA<<<numBlocks, BLOCK_SIZE>>>(d_agents_x, d_agents_y, d_destination_x, d_destination_y,
                                      d_destination_r, d_current_waypoint, d_waypoint_start_offset,
                                      d_waypoints, n_agents);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // TODO: Maybe do an async memcpy and check that it finished before modifying
  // d_agents?
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

/************************************************************************************************
 * Heatmap CUDA implementation *****************************************************************
 * **********************************************************************************************
 */

#define WEIGHTSUM 273

int* d_heatmap;

int* h_blurred_heatmap;
int* d_blurred_heatmap;

int* d_scaled_heatmap;

int* d_desired_pos_x;
int* d_desired_pos_y;

/*WEIGHTS MATRIX FOR GAUSSIAN BLUR*/
__constant__ int w[5][5] = {
    {1, 4, 7, 4, 1}, {4, 16, 26, 16, 4}, {7, 26, 41, 26, 7}, {4, 16, 26, 16, 4}, {1, 4, 7, 4, 1}};

__constant__ int w_row[5] = {1, 4, 7, 4, 1};
__constant__ int w_col[5] = {1, 4, 7, 4, 1};

// __global__ void paintRed(int* heatmap) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int x = idx % SIZE;
//   int y = idx / SIZE;
//   if (idx < SIZE * SIZE && x == 10) {
//     heatmap[idx] = 255;
//   }
// }

/**********************************************
SETUP AND CLEANING FUNCTONALITY FOR HEATMAP
***********************************************/

void Ped::Model::setupHeatmapCUDA() {
  cout << "Setting up heatmap in GPU" << endl;

  // h_blurred_heatmap = (int*)malloc(SCALED_SIZE * SCALED_SIZE * sizeof(int));
  // Do allocation on cpu pinned memory
  cudaHostAlloc(&h_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaHostAllocDefault);

  blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
  for (int i = 0; i < SCALED_SIZE; i++) {
    blurred_heatmap[i] = h_blurred_heatmap + SCALED_SIZE * i;
  }

  int n_agents = agents_soa->getNumAgents();
  cudaMalloc(&d_desired_pos_x, n_agents * sizeof(int));
  cudaMalloc(&d_desired_pos_y, n_agents * sizeof(int));

  cudaMalloc(&d_heatmap, SIZE_GPU * sizeof(int));

  cudaMalloc(&d_scaled_heatmap, SCALED_SIZE_GPU * sizeof(int));
  cudaMalloc(&d_blurred_heatmap, SCALED_SIZE_GPU * sizeof(int));

  // Initialize device memory to zero
  cudaMemset(d_heatmap, 0, SIZE_GPU * sizeof(int));
  cudaMemset(d_scaled_heatmap, 0, SCALED_SIZE_GPU * sizeof(int));
  cudaMemset(d_blurred_heatmap, 0, SCALED_SIZE_GPU * sizeof(int));
}

void Ped::Model::freeHeatmapCUDA() {
  cout << "Cleaning up heatmap in GPU" << endl;

  // Free host memory
  free(blurred_heatmap);

  // Free pinned host memory
  cudaFreeHost(h_blurred_heatmap);

  // Free device memory
  cudaFree(d_heatmap);
  cudaFree(d_scaled_heatmap);
  cudaFree(d_blurred_heatmap);

  cudaFree(d_desired_pos_x);
  cudaFree(d_desired_pos_y);
}

/**********************************************
 * KERNELS FOR HEATMAP
 * *********************************************/

// __global__ void traductor(int* heatmap) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < SCALED_SIZE * SCALED_SIZE) {
//     heatmap[idx] = 0x00FF0000 | heatmap[idx] << 24;
//   }
// }

__global__ void fadeHeatmap(int* heatmap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < SIZE_GPU) {
    heatmap[idx] *= 0.80;
  }
}

__global__ void updateHeatmap(const int* pos_x, const int* pos_y, int* heatmap, int n_agents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_agents) {
    int x = pos_x[idx];
    int y = pos_y[idx];

    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
      return;
    }

    // TODO: check if the value of heatmap is already 255 -> nothing to do
    atomicAdd(&heatmap[y * SIZE + x], 40);
  }
}

__global__ void normScaleHeatmap(int* orig_heatmap, int* scaled_heatmap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < SIZE * SIZE) {
    int value = min(orig_heatmap[idx], 255);
    orig_heatmap[idx] = value;

    int x = idx % SIZE;
    int y = idx / SIZE;

    for (int cellY = 0; cellY < CELLSIZE; cellY++) {
      for (int cellX = 0; cellX < CELLSIZE; cellX++) {
        scaled_heatmap[(y * CELLSIZE + cellY) * SCALED_SIZE + (x * CELLSIZE + cellX)] = value;
      }
    }
  }
}

__global__ void gaussianBlur(int* scaled_heatmap, int* blurred_heatmap) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO: Test if 1D shared_mem array is faster
  __shared__ int shared_mem_tile[BLOCK_SIZE_1D][BLOCK_SIZE_1D];

  // Load data to shared memory (only load data for threads that will output a value)
  if (col < SCALED_SIZE && row < SCALED_SIZE) {
    shared_mem_tile[threadIdx.y][threadIdx.x] = scaled_heatmap[row * SCALED_SIZE + col];
  } else {
    shared_mem_tile[threadIdx.y][threadIdx.x] = 0;
  }

  __syncthreads();

  // Calculate the output value
  if (col < SCALED_SIZE && row < SCALED_SIZE) {
    int sum = 0;
    for (int i_row = -MASK_RADIUS; i_row <= MASK_RADIUS; i_row++) {
      for (int i_col = -MASK_RADIUS; i_col <= MASK_RADIUS; i_col++) {
        // Check if cell to read is within the bounds of the shared memory array (tile)
        int shared_index_x = threadIdx.x + i_col;
        int shared_index_y = threadIdx.y + i_row;
        if (shared_index_x >= 0 && shared_index_x < BLOCK_SIZE_1D && shared_index_y >= 0 &&
            shared_index_y < BLOCK_SIZE_1D) {
          // 1) Core of the tile ->> Using shared memory
          sum += shared_mem_tile[shared_index_y][shared_index_x] *
                 w[i_row + MASK_RADIUS][i_col + MASK_RADIUS];
        } else {
          // 2) Halo of the tile ->> Using global memory
          int global_index_x = col + i_col;
          int global_index_y = row + i_row;
          // Cells outside scenario are not counted (treated as if they were 0)
          if (global_index_x >= 0 && global_index_x < SCALED_SIZE && global_index_y >= 0 &&
              global_index_y < SCALED_SIZE) {
            sum += scaled_heatmap[(global_index_y)*SCALED_SIZE + global_index_x] *
                   w[i_row + MASK_RADIUS][i_col + MASK_RADIUS];
          }
        }
      }
    }
    int value = sum / WEIGHTSUM;
    blurred_heatmap[row * SCALED_SIZE + col] = 0x00FF0000 | value << 24;
  }
}
/**********************************************
 * UPDATE HEATMAP FUNCTION
 * *********************************************/

/*void Ped::Model::updateHeatmapCUDA() {
  //KERNEL LAUNCH VARS
int blocksPerGrid;
int n_agents = agents_soa->getNumAgents();

//TIMING VARS
cudaEvent_t startTotal, stopTotal;
cudaEventCreate(&startTotal);
cudaEventCreate(&stopTotal);
cudaEventRecord(startTotal);
float ms_fade = 0, ms_update = 0, ms_norm_scale = 0, ms_blur = 0, ms_memcpy_desired_pos_x = 0,
      ms_memcpy_desired_pos_y = 0, ms_memcpy_blurred_heatmap = 0;

//1) Copy agent positions to GPU
const int* h_desired_pos_x = agents_soa->getDesiredPosX();
const int* h_desired_pos_y = agents_soa->getDesiredPosY();

// cudaMemcpy(d_desired_pos_x, h_desired_pos_x, n_agents * sizeof(int), cudaMemcpyHostToDevice);
// cudaCheckError(cudaGetLastError());
MEASURE_KERNEL_EXECUTION(ms_memcpy_desired_pos_x,
                         cudaMemcpy(d_desired_pos_x, h_desired_pos_x, n_agents * sizeof(int),
                                    cudaMemcpyHostToDevice));

// cudaMemcpy(d_desired_pos_y, h_desired_pos_y, n_agents * sizeof(int), cudaMemcpyHostToDevice);
// cudaCheckError(cudaGetLastError());
MEASURE_KERNEL_EXECUTION(ms_memcpy_desired_pos_y,
                         cudaMemcpy(d_desired_pos_y, h_desired_pos_y, n_agents * sizeof(int),
                                    cudaMemcpyHostToDevice));

//2) Launch timed kernels

blocksPerGrid = (SIZE * SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
MEASURE_KERNEL_EXECUTION(ms_fade, fadeHeatmap<<<blocksPerGrid, BLOCK_SIZE>>>(d_heatmap));

blocksPerGrid = (n_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
MEASURE_KERNEL_EXECUTION(ms_update, updateHeatmap<<<blocksPerGrid, BLOCK_SIZE>>>(
                                        d_desired_pos_x, d_desired_pos_y, d_heatmap, n_agents));

blocksPerGrid = (SIZE * SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
MEASURE_KERNEL_EXECUTION(ms_norm_scale,
                         normScaleHeatmap<<<blocksPerGrid, BLOCK_SIZE>>>(d_heatmap,
                                                                         d_scaled_heatmap));

// Timing for gaussianBlur kernel
dim3 blockDim(BLOCK_SIZE_1D, BLOCK_SIZE_1D);
dim3 gridDim((SCALED_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D,
             (SCALED_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);

MEASURE_KERNEL_EXECUTION(ms_blur,
                         gaussianBlur<<<gridDim, blockDim>>>(d_scaled_heatmap, d_blurred_heatmap));
//3) Copy back blurred heatmap to host
// cudaMemcpy(h_blurred_heatmap, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int),
// cudaMemcpyDeviceToHost); cudaCheckError(cudaGetLastError());
MEASURE_KERNEL_EXECUTION(ms_memcpy_blurred_heatmap,
                         cudaMemcpy(h_blurred_heatmap, d_blurred_heatmap,
                                    SCALED_SIZE* SCALED_SIZE * sizeof(int),
                                    cudaMemcpyDeviceToHost));

//4) Finish GPU Timing
cudaEventRecord(stopTotal);
cudaEventSynchronize(stopTotal);
float millisecondsTotal = 0;
cudaEventElapsedTime(&millisecondsTotal, startTotal, stopTotal);

//5) Update timing vars
total_ms_fade += ms_fade;
total_ms_update += ms_update;
total_ms_norm_scale += ms_norm_scale;
total_ms_blur += ms_blur;
total_ms_memcpy_desired_pos_x += ms_memcpy_desired_pos_x;
total_ms_memcpy_desired_pos_y += ms_memcpy_desired_pos_y;
total_ms_memcpy_blurred_heatmap += ms_memcpy_blurred_heatmap;
total_time_GPU += millisecondsTotal;

// Clean up cuda events
cudaEventDestroy(startTotal);
cudaEventDestroy(stopTotal);
}
*/

void Ped::Model::copyDesiredPosToGPU() {
  int n_agents = agents_soa->getNumAgents();

  // 1) Copy agent positions to GPU
  const int* h_desired_pos_x = agents_soa->getDesiredPosX();
  const int* h_desired_pos_y = agents_soa->getDesiredPosY();

  // cudaMemcpy(d_desired_pos_x, h_desired_pos_x, n_agents * sizeof(int), cudaMemcpyHostToDevice);
  // cudaCheckError(cudaGetLastError());
  cudaMemcpy(d_desired_pos_x, h_desired_pos_x, n_agents * sizeof(int), cudaMemcpyHostToDevice);

  // cudaMemcpy(d_desired_pos_y, h_desired_pos_y, n_agents * sizeof(int), cudaMemcpyHostToDevice);
  // cudaCheckError(cudaGetLastError());
  cudaMemcpy(d_desired_pos_y, h_desired_pos_y, n_agents * sizeof(int), cudaMemcpyHostToDevice);
}

void Ped::Model::updateHeatmapCUDA() {
  // KERNEL LAUNCH VARS
  int blocksPerGrid;
  int n_agents = agents_soa->getNumAgents();

  // TIMING VARS
  cudaEvent_t startTotal, stopTotal;
  cudaEventCreate(&startTotal);
  cudaEventCreate(&stopTotal);
  cudaEventRecord(startTotal);

  // 2) Launch timed kernels
  blocksPerGrid = (SIZE * SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fadeHeatmap<<<blocksPerGrid, BLOCK_SIZE>>>(d_heatmap);

  blocksPerGrid = (n_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
  updateHeatmap<<<blocksPerGrid, BLOCK_SIZE>>>(d_desired_pos_x, d_desired_pos_y, d_heatmap,
                                               n_agents);

  blocksPerGrid = (SIZE * SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  normScaleHeatmap<<<blocksPerGrid, BLOCK_SIZE>>>(d_heatmap, d_scaled_heatmap);

  // Timing for gaussianBlur kernel
  dim3 blockDim(BLOCK_SIZE_1D, BLOCK_SIZE_1D);
  dim3 gridDim((SCALED_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D,
               (SCALED_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);

  gaussianBlur<<<gridDim, blockDim>>>(d_scaled_heatmap, d_blurred_heatmap);
  // 3) Copy back blurred heatmap to host
  //  cudaMemcpy(h_blurred_heatmap, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int),
  //  cudaMemcpyDeviceToHost); cudaCheckError(cudaGetLastError());
  cudaMemcpy(h_blurred_heatmap, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int),
             cudaMemcpyDeviceToHost);

  // 4) Finish GPU Timing
  cudaEventRecord(stopTotal);
  cudaEventSynchronize(stopTotal);
  float millisecondsTotal = 0;
  cudaEventElapsedTime(&millisecondsTotal, startTotal, stopTotal);

  // 5) Update timing vars
  total_time_GPU += millisecondsTotal;

  // Clean up cuda events
  cudaEventDestroy(startTotal);
  cudaEventDestroy(stopTotal);
}

// function to print the average time taken by each kernel and memory copy and the total time
void Ped::Model::print_gpu_heatmap_avg_timings(int n_steps) {
  float avg_fade = total_ms_fade / n_steps;
  float avg_update = total_ms_update / n_steps;
  float avg_norm_scale = total_ms_norm_scale / n_steps;
  float avg_blur = total_ms_blur / n_steps;
  float avg_memcpy_desired_pos_x = total_ms_memcpy_desired_pos_x / n_steps;
  float avg_memcpy_desired_pos_y = total_ms_memcpy_desired_pos_y / n_steps;
  float avg_memcpy_blurred_heatmap = total_ms_memcpy_blurred_heatmap / n_steps;

  float avg_total = total_time_GPU / n_steps;

  float avg_tick = total_tick_time / n_steps;
  float avg_diff = total_diff / n_steps;

  // Kernel_name, avg_time (ms)
  cout << "Kernel_name, avg_time (ms)" << endl;
  cout << "fadeHeatmap, " << avg_fade << endl;
  cout << "updateHeatmap, " << avg_update << endl;
  cout << "normScaleHeatmap, " << avg_norm_scale << endl;
  cout << "gaussianBlur, " << avg_blur << endl;
  cout << "memcpy_desired_pos_x, " << avg_memcpy_desired_pos_x << endl;
  cout << "memcpy_desired_pos_y, " << avg_memcpy_desired_pos_y << endl;
  cout << "memcpy_blurred_heatmap, " << avg_memcpy_blurred_heatmap << endl;
  cout << "total_time_GPU, " << avg_total << endl;
  cout << "tick_time_CPU, " << avg_tick << endl;
  cout << "diff_time, " << avg_diff << endl;

  // Kernel_name, avg_time (ms)
}
