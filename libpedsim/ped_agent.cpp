//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//
#include "ped_agent.h"

#include <math.h>

#include "ped_waypoint.h"

Ped::Tagent::Tagent(int posX, int posY) { Ped::Tagent::init(posX, posY); }

Ped::Tagent::Tagent(double posX, double posY) {
  Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY) {
  x = posX;
  y = posY;
  destination = NULL;
  // This isn't used
  lastDestination = NULL;
}

void Ped::Tagent::computeNextDesiredPosition() {
  destination = getNextDestination();
  if (destination == NULL) {
    return;
  }

  auto x_diff = destination->getx() - x;
  auto y_diff = destination->gety() - y;

  auto length = sqrt((x_diff * x_diff) + (y_diff * y_diff));
  auto next_x = x + round(x_diff / length);
  auto next_y = y + round(y_diff / length);

  desiredPositionX = next_x;
  desiredPositionY = next_y;
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) { waypoints.push_back(wp); }

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
  Ped::Twaypoint* nextDestination = NULL;
  bool agentReachedDestination = false;

  if (destination != NULL) {
    // compute if agent reached its current destination
    double diffX = destination->getx() - x;
    double diffY = destination->gety() - y;
    double length = sqrt(diffX * diffX + diffY * diffY);
    agentReachedDestination = length < destination->getr();
  }

  if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
    // Case 1: agent has reached destination (or has no current destination);
    // get next destination if available
    waypoints.push_back(destination);
    nextDestination = waypoints.front();
    waypoints.pop_front();
  } else {
    // Case 2: agent has not yet reached destination, continue to move towards
    // current destination
    nextDestination = destination;
  }

  return nextDestination;
}
