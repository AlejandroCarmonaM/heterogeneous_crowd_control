#pragma once

#include <atomic>

#include "utils.h"

class CollisionChecker {
 public:
  CollisionChecker(int* agents_x, int* agents_y, int n_agents, Waypoint* waypoints,
                   int total_waypoints);
  ~CollisionChecker();

  // Struct to store dimensions of the checked area
  struct AreaLimits {
    int x;
    int y;
    int w;
    int h;
  };
  AreaLimits getColAreaLimits() { return {x, y, w, h}; }

  bool isInArea(int pos_x, int pos_y);
  void resizeArea();

  bool isNearBorder(int pos_x, int pos_y);
  int getQuadrant(int pos_x, int pos_y);

  bool simpleCheckMove(int pos_x1, int pos_y1, int pos_x2, int pos_y2);
  bool borderCheckMove(int pos_x1, int pos_y1, int pos_x2, int pos_y2);

 private:
  static const int AREA_PADDING;

  int x, y;  // Highest leftmost point of the rectangular area
  int w, h;  // Width and height of the area
  int midx, midy;

  // ref to the agents_soa agents_x and agents_y
  int* agents_x;
  int* agents_y;
  int n_agents;

  // Bool matrix
  std::atomic_bool* positions;

  // For easy access to positions
  std::atomic_bool& isTaken(int pos_x, int pos_y);

  void setLimits(Waypoint* waypoints, int total_waypoints);
};