#pragma once

#include <set>
#include <thread>

#include "collision_checker.h"
#include "ped_agent.h"
#include "utils.h"

namespace Ped {
class Model;

class TagentSoA {
 public:
  static constexpr int MAX_THREADS = 16;
  static constexpr int NUM_QUADRANTS = 4;

  TagentSoA(std::vector<Ped::Tagent*> agents, bool heatmap_cuda);
  ~TagentSoA();

  void setupPthreads(int n_threads);
  void freePthreads();
  void setupCUDA();
  void freeCUDA();
  void setupColCheckSeq();
  void setupColCheckPar();

  void seqTick();
  void ompTick();
  void pthreadTick();
  void vectorTick();
  void callTickCUDA();
  void colPreventSeqTick();
  void colPreventParTick();

  int getNumAgents() { return n_agents; }
  int* getAgentsX() { return agents_x; }
  int* getAgentsY() { return agents_y; }
  int* getDesiredPosX() { return desired_pos_x; }
  int* getDesiredPosY() { return desired_pos_y; }

  // Prints to stdout all found collisions, returns true if there's at least one collision
  bool printCollisions();

  CollisionChecker::AreaLimits getColAreaLimits() { return col_checker->getColAreaLimits(); }

 private:
  static constexpr int AGENTS_PER_VECTOR = 4;
  static constexpr int N_ALTERNATIVES = 3;

  struct CrossingAgent {
    int agent;
    int dest_quadrant;
  };

  struct OutOfBoundsMovement {
    int agent;
    int src_x, src_y;
    std::vector<std::pair<int, int>> alternatives;
  };

  int n_agents;
  int total_waypoints;

  // SoA
  int* agents_x = nullptr;
  int* agents_y = nullptr;
  int* destination_x = nullptr;
  int* destination_y = nullptr;
  int* destination_r = nullptr;
  int* desired_pos_x = nullptr;
  int* desired_pos_y = nullptr;
  int* current_waypoint = nullptr;
  int* waypoint_start_offset = nullptr;

  Waypoint* waypoints = nullptr;

  // pthreads
  int n_threads;
  int main_thread_start;
  std::thread workers[MAX_THREADS - 1];
  std::atomic<bool> exit_flag;
  pthread_barrier_t barrier;
  bool pthreadWorkerTick(int start, int end);

  CollisionChecker* col_checker = nullptr;
  std::set<int>* quadrant_agents = nullptr;

  void setNewDestination(int i, int num_waypoints);

  std::pair<int, int> getMovement(int i);

  void processQuadrant(int quadrant, std::vector<CrossingAgent>& crossing_agents,
                       std::vector<OutOfBoundsMovement>& out_of_bounds_agents);

  // boolean var to check if heatmap has been run on CUDA
  bool m_heatmap_cuda = false;
};

}  // namespace Ped