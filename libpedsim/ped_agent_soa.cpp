#include "ped_agent_soa.h"

#include <cuda_runtime.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <thread>

#include "ped_model.h"
#include "ped_waypoint.h"
#include "tick_cuda.h"

Ped::TagentSoA::TagentSoA(std::vector<Ped::Tagent*> agents, bool heatmap_cuda) {
  constexpr int ALIGNMENT = AGENTS_PER_VECTOR * sizeof(int);

  n_agents = agents.size();
  m_heatmap_cuda = heatmap_cuda;

  agents_x = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));
  agents_y = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));

  destination_x = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));
  destination_y = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));
  destination_r = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));

  // declare desired_pos_x and desired_pos_y as CUDA pinned memory (if needed)
  if (m_heatmap_cuda) {
    cudaMallocHost((void**)&desired_pos_x, n_agents * sizeof(int));
    cudaMallocHost((void**)&desired_pos_y, n_agents * sizeof(int));
    // declare desired_pos_x and desired_pos_y as normal memory if heatmap not on CUDA
  } else {
    desired_pos_x = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));
    desired_pos_y = static_cast<int*>(std::aligned_alloc(ALIGNMENT, n_agents * sizeof(int)));
  }

  current_waypoint = new int[n_agents];

  // This +1 is necessary because each agent knows the length of its waypoint array by looking at
  // the start of the next array (length_array[i] = start[i+1]-start[i]), and we need to set some
  // value for the agent after the last
  waypoint_start_offset = new int[n_agents + 1];

  total_waypoints = 0;
  for (int i = 0; i < n_agents; i++) {
    int n = agents[i]->getNumWaypoints();
    if (n == 0) {
      // Agents with no waypoints will be assigned a default one for easier processing
      n = 1;
    }

    total_waypoints += n;
  }

  waypoints = new Waypoint[total_waypoints];

  int waypoint_offset = 0;
  for (int i = 0; i < n_agents; i++) {
    Ped::Tagent* agent = agents[i];
    agents_x[i] = agent->getX();
    agents_y[i] = agent->getY();

    waypoint_start_offset[i] = waypoint_offset;

    int n_waypoints_agent = agents[i]->getNumWaypoints();

    if (n_waypoints_agent == 0) {
      // Create a default waypoint with the agent's position so we can always assign an initial
      // destination to agents
      waypoints[waypoint_offset] = Waypoint{agents_x[i], agents_y[i], 0};
      n_waypoints_agent = 1;
    } else {
      auto agent_waypoints = agent->getWaypoints();

      // Move waypoints from deque to array
      for (int j = 0; j < n_waypoints_agent; j++) {
        Waypoint* w = &waypoints[waypoint_offset + j];
        w->x = agent_waypoints->at(j)->getx();
        w->y = agent_waypoints->at(j)->gety();
        w->r = agent_waypoints->at(j)->getr();
      }
    }

    current_waypoint[i] = -1;
    // setNewDestination makes current_waypoint be 0 (the first waypoint)
    setNewDestination(i, n_waypoints_agent);

    waypoint_offset += n_waypoints_agent;
  }

  // Set an extra waypoint offset so that the last agent knows where its waypoint array ends
  waypoint_start_offset[n_agents] =
      waypoint_start_offset[n_agents - 1] + agents[n_agents - 1]->getNumWaypoints();
}

void Ped::TagentSoA::setupPthreads(int n_threads) {
  this->n_threads = n_threads;
  exit_flag.store(false);
  pthread_barrier_init(&barrier, nullptr, n_threads);

  int start = 0;
  int remaining_agents = n_agents % n_threads;

  // Create n_threads-1 threads
  for (int i = 0; i < n_threads - 1; i++) {
    // First remaining_agents threads will have 1 more agent than the rest
    int chunk_size = n_agents / n_threads + (i < remaining_agents ? 1 : 0);
    int end = start + chunk_size;

    // Created threads will wait each tick at barrier in pthreadTask for main thread
    workers[i] = std::thread([this, start, end]() {
      while (pthreadWorkerTick(start, end));
    });
    start = end;
  }
  main_thread_start = start;
}

void Ped::TagentSoA::setupColCheckSeq() {
  col_checker = new CollisionChecker(agents_x, agents_y, n_agents, waypoints, total_waypoints);
}

void Ped::TagentSoA::setupColCheckPar() {
  col_checker = new CollisionChecker(agents_x, agents_y, n_agents, waypoints, total_waypoints);

  quadrant_agents = new std::set<int>[NUM_QUADRANTS];
  for (int i = 0; i < n_agents; i++) {
    int quadrant = col_checker->getQuadrant(agents_x[i], agents_y[i]);
    quadrant_agents[quadrant].insert(i);
  }
}

void Ped::TagentSoA::setNewDestination(int agent, int num_waypoints) {
  current_waypoint[agent] = (current_waypoint[agent] + 1) % num_waypoints;

  Waypoint* w = &(waypoints[waypoint_start_offset[agent] + current_waypoint[agent]]);

  destination_x[agent] = w->x;
  destination_y[agent] = w->y;
  destination_r[agent] = w->r;
}

std::pair<int, int> Ped::TagentSoA::getMovement(int agent) {
  // Check if agent reached its current destination
  // Agents always have a set destination (they are initialized with one)
  double diff_x = destination_x[agent] - agents_x[agent];
  double diff_y = destination_y[agent] - agents_y[agent];
  double length = sqrt(diff_x * diff_x + diff_y * diff_y);
  bool agentReachedDestination = length < destination_r[agent];

  if (agentReachedDestination) {
    int num_waypoints = waypoint_start_offset[agent + 1] - waypoint_start_offset[agent];
    // Only change destination if agent has more than 1 total waypoints
    if (num_waypoints > 1) {
      setNewDestination(agent, num_waypoints);

      // Update diff and length to new destination
      diff_x = destination_x[agent] - agents_x[agent];
      diff_y = destination_y[agent] - agents_y[agent];
      length = sqrt(diff_x * diff_x + diff_y * diff_y);
    }
  }

  // Round movement to nearest integer
  return {round(diff_x / length), round(diff_y / length)};
}

void Ped::TagentSoA::seqTick() {
  for (int i = 0; i < n_agents; i++) {
    std::pair<int, int> movement = getMovement(i);
    desired_pos_x[i] = agents_x[i] + movement.first;
    desired_pos_y[i] = agents_y[i] + movement.second;

    agents_x[i] = desired_pos_x[i];
    agents_y[i] = desired_pos_y[i];
  }
}

void Ped::TagentSoA::ompTick() {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_agents; i++) {
    std::pair<int, int> movement = getMovement(i);
    desired_pos_x[i] = agents_x[i] + movement.first;
    desired_pos_y[i] = agents_y[i] + movement.second;

    agents_x[i] = desired_pos_x[i];
    agents_y[i] = desired_pos_y[i];
  }
}

bool Ped::TagentSoA::pthreadWorkerTick(int start, int end) {
  // Ensure workers wait for the main thread
  pthread_barrier_wait(&barrier);

  if (exit_flag.load()) {
    return false;
  }

  for (int i = start; i < end; i++) {
    std::pair<int, int> movement = getMovement(i);
    desired_pos_x[i] = agents_x[i] + movement.first;
    desired_pos_y[i] = agents_y[i] + movement.second;

    agents_x[i] = desired_pos_x[i];
    agents_y[i] = desired_pos_y[i];
  }

  // Ensure main thread only exits after all workers have finished their work
  pthread_barrier_wait(&barrier);

  return true;
}

void Ped::TagentSoA::pthreadTick() { pthreadWorkerTick(main_thread_start, n_agents); }

void Ped::TagentSoA::freePthreads() {
  exit_flag.store(true);
  pthread_barrier_wait(&barrier);
  for (int i = 0; i < n_threads - 1; i++) {
    workers[i].join();
  }
}

void Ped::TagentSoA::vectorTick() {
#pragma omp parallel for schedule(static)
  for (int i = 0; i <= n_agents - AGENTS_PER_VECTOR; i += AGENTS_PER_VECTOR) {
    __m128i agents_x_vec = _mm_load_si128((__m128i*)&agents_x[i]);
    __m128i agents_y_vec = _mm_load_si128((__m128i*)&agents_y[i]);

    __m128i destination_x_vec = _mm_load_si128((__m128i*)&destination_x[i]);
    __m128i destination_y_vec = _mm_load_si128((__m128i*)&destination_y[i]);

    // Check if agent reached its current destination
    // Agents always have a set destination (they are initialized with one)
    __m256d diffX_vec = _mm256_cvtepi32_pd(_mm_sub_epi32(destination_x_vec, agents_x_vec));
    __m256d diffY_vec = _mm256_cvtepi32_pd(_mm_sub_epi32(destination_y_vec, agents_y_vec));
    __m256d diffX_sq_vec = _mm256_mul_pd(diffX_vec, diffX_vec);
    __m256d diffY_sq_vec = _mm256_mul_pd(diffY_vec, diffY_vec);
    __m256d length_vec = _mm256_sqrt_pd(_mm256_add_pd(diffX_sq_vec, diffY_sq_vec));

    // Sequentially change destination if reached
    for (int vec_index = 0; vec_index < AGENTS_PER_VECTOR; vec_index++) {
      int index = i + vec_index;
      double* length_vec_ptr = (double*)&length_vec;
      double* diffX_vec_ptr = (double*)&diffX_vec;
      double* diffY_vec_ptr = (double*)&diffY_vec;

      bool agentReachedDestination = length_vec_ptr[vec_index] < destination_r[index];

      if (agentReachedDestination) {
        int num_waypoints = waypoint_start_offset[index + 1] - waypoint_start_offset[index];
        // Only change destination if agent has more than 1 total waypoints
        if (num_waypoints > 1) {
          // Set destination to next waypoint
          setNewDestination(index, num_waypoints);

          // Update length to new destination
          int new_diffx = (double)(destination_x[index] - agents_x[index]);
          int new_diffy = (double)(destination_y[index] - agents_y[index]);
          diffX_vec_ptr[vec_index] = new_diffx;
          diffY_vec_ptr[vec_index] = new_diffy;
          length_vec_ptr[vec_index] = sqrt(new_diffx * new_diffx + new_diffy * new_diffy);
        }
      }
    }

    __m256d x_move = _mm256_div_pd(diffX_vec, length_vec);
    __m256d y_move = _mm256_div_pd(diffY_vec, length_vec);

    // Round movement to nearest integer
    agents_x_vec = _mm_add_epi32(
        agents_x_vec, _mm256_cvtpd_epi32(_mm256_round_pd(x_move, _MM_FROUND_TO_NEAREST_INT)));
    agents_y_vec = _mm_add_epi32(
        agents_y_vec, _mm256_cvtpd_epi32(_mm256_round_pd(y_move, _MM_FROUND_TO_NEAREST_INT)));

    // Move agent
    _mm_store_si128((__m128i*)&agents_x[i], agents_x_vec);
    _mm_store_si128((__m128i*)&agents_y[i], agents_y_vec);

    _mm_store_si128((__m128i*)&desired_pos_x[i], agents_x_vec);
    _mm_store_si128((__m128i*)&desired_pos_y[i], agents_y_vec);
  }

  // Process remaining agents sequentially
  int remaining_agents = n_agents % AGENTS_PER_VECTOR;
  for (int i = n_agents - remaining_agents; i < n_agents; i++) {
    std::pair<int, int> movement = getMovement(i);
    desired_pos_x[i] = agents_x[i] + movement.first;
    desired_pos_y[i] = agents_y[i] + movement.second;

    agents_x[i] = desired_pos_x[i];
    agents_y[i] = desired_pos_y[i];
  }
}

void Ped::TagentSoA::colPreventSeqTick() {
  std::pair<int, int> alternatives[N_ALTERNATIVES];

  for (int i = 0; i < n_agents; i++) {
    // Compute alternative positions that would bring the agent closer to his desiredPosition,
    // starting with the desiredPosition itself
    std::pair<int, int> desired_move(getMovement(i));
    std::pair<int, int> desired_pos(agents_x[i] + desired_move.first,
                                    agents_y[i] + desired_move.second);

    desired_pos_x[i] = desired_pos.first;
    desired_pos_y[i] = desired_pos.second;

    int diff_x = desired_move.first;
    int diff_y = desired_move.second;

    alternatives[0] = desired_pos;
    if (diff_x == 0 || diff_y == 0) {
      // Agent wants to walk straight North, South, West or East
      alternatives[1] = std::make_pair(desired_pos.first + diff_y, desired_pos.second + diff_x);
      alternatives[2] = std::make_pair(desired_pos.first - diff_y, desired_pos.second - diff_x);
    } else {
      // Agent wants to walk diagonally (e.g. diffX = 1, diffY = 1)
      alternatives[1] = std::make_pair(desired_pos.first, agents_y[i]);
      alternatives[2] = std::make_pair(agents_x[i], desired_pos.second);
    }

    // Find the first empty alternative position and move there
    for (int j = 0; j < N_ALTERNATIVES; j++) {
      std::pair<int, int> alternative = alternatives[j];
      bool is_valid = col_checker->simpleCheckMove(agents_x[i], agents_y[i], alternative.first,
                                                   alternative.second);

      if (is_valid) {
        agents_x[i] = alternative.first;
        agents_y[i] = alternative.second;
        break;
      }
    }
  }
}

void Ped::TagentSoA::processQuadrant(int quadrant, std::vector<CrossingAgent>& crossing_agents,
                                     std::vector<OutOfBoundsMovement>& out_of_bounds_agents) {
  std::pair<int, int> alternatives[N_ALTERNATIVES];

  for (int i : quadrant_agents[quadrant]) {
    // Compute alternative positions that would bring the agent closer to his desiredPosition,
    // starting with the desiredPosition itself
    std::pair<int, int> desired_move(getMovement(i));
    std::pair<int, int> desired_pos(agents_x[i] + desired_move.first,
                                    agents_y[i] + desired_move.second);

    desired_pos_x[i] = desired_pos.first;
    desired_pos_y[i] = desired_pos.second;

    int diff_x = desired_move.first;
    int diff_y = desired_move.second;

    alternatives[0] = desired_pos;
    if (diff_x == 0 || diff_y == 0) {
      // Agent wants to walk straight North, South, West or East
      alternatives[1] = std::make_pair(desired_pos.first + diff_y, desired_pos.second + diff_x);
      alternatives[2] = std::make_pair(desired_pos.first - diff_y, desired_pos.second - diff_x);
    } else {
      // Agent wants to walk diagonally (e.g. diffX = 1, diffY = 1)
      alternatives[1] = std::make_pair(desired_pos.first, agents_y[i]);
      alternatives[2] = std::make_pair(agents_x[i], desired_pos.second);
    }

    // Find the first empty alternative position and move there
    for (int j = 0; j < N_ALTERNATIVES; j++) {
      std::pair<int, int> alternative = alternatives[j];

      if (!col_checker->isInArea(alternative.first, alternative.second)) {
        std::vector<std::pair<int, int>> remaining_alts;
        for (int k = j; k < N_ALTERNATIVES; k++) {
          remaining_alts.push_back(alternatives[k]);
        }
        out_of_bounds_agents.push_back({i, agents_x[i], agents_y[i], remaining_alts});
        break;
      }

      bool near_border = col_checker->isNearBorder(alternative.first, alternative.second);

      if (near_border) {
        bool is_valid = col_checker->borderCheckMove(agents_x[i], agents_y[i], alternative.first,
                                                     alternative.second);

        if (is_valid) {
          agents_x[i] = alternative.first;
          agents_y[i] = alternative.second;

          int dest_quadrant = col_checker->getQuadrant(alternative.first, alternative.second);
          if (dest_quadrant != quadrant) {
            crossing_agents.push_back(CrossingAgent{i, dest_quadrant});
          }
          break;
        }
      } else {
        bool is_valid = col_checker->simpleCheckMove(agents_x[i], agents_y[i], alternative.first,
                                                     alternative.second);

        if (is_valid) {
          agents_x[i] = alternative.first;
          agents_y[i] = alternative.second;

          break;
        }
      }
    }
  }
}

void Ped::TagentSoA::colPreventParTick() {
  std::vector<CrossingAgent> crossing_agents_quadrant[NUM_QUADRANTS];
  std::vector<OutOfBoundsMovement> out_of_bounds_movements[NUM_QUADRANTS];

#pragma omp parallel
  {
#pragma omp single nowait
    {
      for (int quadrant = 0; quadrant < NUM_QUADRANTS; quadrant++) {
#pragma omp task
        {
          processQuadrant(quadrant, crossing_agents_quadrant[quadrant],
                          out_of_bounds_movements[quadrant]);
        }
      }
    }
#pragma omp taskwait
  }

  // Move agents that are out of bounds
  for (int quadrant = 0; quadrant < NUM_QUADRANTS; quadrant++) {
    for (auto oob_mov : out_of_bounds_movements[quadrant]) {
      for (auto alternative : oob_mov.alternatives) {
        bool is_valid = col_checker->simpleCheckMove(oob_mov.src_x, oob_mov.src_y,
                                                     alternative.first, alternative.second);
        if (is_valid) {
          agents_x[oob_mov.agent] = alternative.first;
          agents_y[oob_mov.agent] = alternative.second;

          int dest_quadrant = col_checker->getQuadrant(alternative.first, alternative.second);
          if (dest_quadrant != quadrant) {
            crossing_agents_quadrant[quadrant].push_back(
                CrossingAgent{oob_mov.agent, dest_quadrant});
          }
          break;
        }
      }
    }
  }

  // Assign agents that are crossing quadrants to list of dst quadrant
  for (int quadrant = 0; quadrant < NUM_QUADRANTS; quadrant++) {
    for (auto cross_info : crossing_agents_quadrant[quadrant]) {
      quadrant_agents[quadrant].erase(cross_info.agent);
      quadrant_agents[cross_info.dest_quadrant].insert(cross_info.agent);
    }
  }
}

Ped::TagentSoA::~TagentSoA() {
  // std::free for aligned memory
  if (agents_x) {
    std::free(agents_x);
    agents_x = nullptr;
  }
  if (agents_y) {
    std::free(agents_y);
    agents_y = nullptr;
  }

  if (destination_x) {
    std::free(destination_x);
    destination_x = nullptr;
  }
  if (destination_y) {
    std::free(destination_y);
    destination_y = nullptr;
  }
  if (destination_r) {
    std::free(destination_r);
    destination_r = nullptr;
  }
  if (desired_pos_x) {
    if (m_heatmap_cuda) {
      // free desired_pos_x and desired_pos_y as CUDA pinned memory
      cudaFreeHost(desired_pos_x);
      desired_pos_x = nullptr;
    } else {
      std::free(desired_pos_x);
      desired_pos_x = nullptr;
    }
  }
  if (desired_pos_y) {
    if (m_heatmap_cuda) {
      // free desired_pos_x and desired_pos_y as CUDA pinned memory
      cudaFreeHost(desired_pos_y);
      desired_pos_y = nullptr;
    } else {
      std::free(desired_pos_y);
      desired_pos_y = nullptr;
    }
  }

  // Normal delete
  if (current_waypoint) {
    delete[] current_waypoint;
    current_waypoint = nullptr;
  }
  if (waypoint_start_offset) {
    delete[] waypoint_start_offset;
    waypoint_start_offset = nullptr;
  }
  if (waypoints) {
    delete[] waypoints;
    waypoints = nullptr;
  }

  if (col_checker) {
    delete col_checker;
    col_checker = nullptr;
  }
  if (quadrant_agents) {
    delete[] quadrant_agents;
    quadrant_agents = nullptr;
  }
}

bool Ped::TagentSoA::printCollisions() {
  std::set<std::pair<int, int>> seen;
  bool collision_found = false;

  for (int i = 0; i < n_agents; i++) {
    std::pair<int, int> position = {agents_x[i], agents_y[i]};
    if (seen.find(position) != seen.end()) {
      printf("Collision found in [%d:%d]\n", position.first, position.second);
      collision_found = true;
    }
    seen.insert(position);
  }

  return collision_found;
}
