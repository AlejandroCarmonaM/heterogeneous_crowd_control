#include "collision_checker.h"

#include <climits>
#include <iostream>
#include <vector>

const int CollisionChecker::AREA_PADDING = 20;

CollisionChecker::CollisionChecker(int* agents_x, int* agents_y, int n_agents, Waypoint* waypoints,
                                   int total_waypoints)
    : agents_x(agents_x), agents_y(agents_y), n_agents(n_agents) {
  // Initialize limits of the area (x, y, w, h)
  setLimits(waypoints, total_waypoints);

  positions = new std::atomic_bool[w * h];
  for (int i = 0; i < w * h; i++) {
    positions[i] = false;
  }

  for (int i = 0; i < n_agents; i++) {
    isTaken(agents_x[i], agents_y[i]) = true;
  }
}

// Run through waypoints and agents to calculate initial limits of the area
void CollisionChecker::setLimits(Waypoint* waypoints, int total_waypoints) {
  int min_x = INT_MAX;
  int min_y = INT_MAX;
  int max_x = INT_MIN;
  int max_y = INT_MIN;

  for (int i = 0; i < n_agents; i++) {
    min_x = std::min(min_x, agents_x[i]);
    min_y = std::min(min_y, agents_y[i]);
    max_x = std::max(max_x, agents_x[i]);
    max_y = std::max(max_y, agents_y[i]);
  }

  for (int i = 0; i < total_waypoints; i++) {
    min_x = std::min(min_x, waypoints[i].x - waypoints[i].r);
    min_y = std::min(min_y, waypoints[i].y - waypoints[i].r);
    max_x = std::max(max_x, waypoints[i].x + waypoints[i].r);
    max_y = std::max(max_y, waypoints[i].y + waypoints[i].r);
  }

  // Add some padding to the limits
  x = min_x - CollisionChecker::AREA_PADDING;
  y = min_y - CollisionChecker::AREA_PADDING;
  w = max_x - x + 1 + 2 * CollisionChecker::AREA_PADDING;
  h = max_y - y + 1 + 2 * CollisionChecker::AREA_PADDING;

  midx = x + w / 2;
  midy = y + h / 2;

  printf("Setting collision checker limits [%d:%d] w=%d h=%d\n", x, y, w, h);
}

CollisionChecker::~CollisionChecker() { delete[] positions; }

bool CollisionChecker::isNearBorder(int pos_x, int pos_y) {
  // Check if agent is in vertical border
  return ((pos_x == midx - 1 || pos_x == midx) && (pos_y >= y && pos_y < y + h)) ||
         // Check if agent is in horizontal border
         ((pos_x >= x && pos_x < x + w) && (pos_y == midy - 1 || pos_y == midy));
}

int CollisionChecker::getQuadrant(int pos_x, int pos_y) {
  // TODO: Optimize this
  if ((pos_x < midx) && (pos_y < midy)) return 0;
  if ((pos_x >= midx) && (pos_y < midy)) return 1;
  if ((pos_x >= midx) && (pos_y >= midy)) return 2;
  return 3;
}

bool CollisionChecker::simpleCheckMove(int pos_x1, int pos_y1, int pos_x2, int pos_y2) {
  if (!isInArea(pos_x2, pos_y2)) {
    resizeArea();
    // dst can't be taken after resizing, so no need to check

    isTaken(pos_x2, pos_y2) = true;
    isTaken(pos_x1, pos_y1) = false;

    return true;
  } else {
    if (isTaken(pos_x2, pos_y2)) {
      return false;
    } else {
      // isTaken[pos_x2][pos_y2] = true;
      isTaken(pos_x2, pos_y2) = true;
      // isTaken[pos_x1][pos_y1] = false;
      isTaken(pos_x1, pos_y1) = false;

      return true;
    }
  }
}

bool CollisionChecker::borderCheckMove(int pos_x1, int pos_y1, int pos_x2, int pos_y2) {
  if (isTaken(pos_x2, pos_y2)) {
    return false;
  } else {
    // Old position
    std::atomic_bool* addr1 = &isTaken(pos_x1, pos_y1);

    // New position
    std::atomic_bool* addr2 = &isTaken(pos_x2, pos_y2);

    // TODO: Try other versions of compare_exchange and relax memory order of
    // store for better performance
    bool expected = false;
    if (addr2->compare_exchange_strong(expected, true)) {
      // Set old position to not taken
      addr1->store(false);
      return true;
    } else {
      return false;
    }
  }
}

// Add AREA_PADDING in all directions to the dimensions of the current area and the positions array
void CollisionChecker::resizeArea() {
  // 1) Calculate new dimensions
  x = x - AREA_PADDING;
  y = y - AREA_PADDING;
  w = w + 2 * AREA_PADDING;
  h = h + 2 * AREA_PADDING;

  // 2) Delete old positions array
  delete[] positions;

  // 3) Create new positions array
  positions = new std::atomic_bool[w * h]();
  for (int i = 0; i < w * h; i++) {
    positions[i] = false;
  }

  // 4) Run through all the agents to obtain their new positions and update the new positions array
  for (int i = 0; i < n_agents; i++) {
    isTaken(agents_x[i], agents_y[i]) = true;
  }
}

bool CollisionChecker::isInArea(int pos_x, int pos_y) {
  return ((pos_x >= x) && (pos_x < (x + w)) && (pos_y >= y) && (pos_y < (y + h)));
}

std::atomic_bool& CollisionChecker::isTaken(int pos_x, int pos_y) {
  int relative_x = pos_x - x;
  int relative_y = pos_y - y;

  return positions[relative_y * w + relative_x];
}
