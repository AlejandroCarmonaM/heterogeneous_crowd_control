//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp.
#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

// g++ does not understand cuda stuff. This makes it ignore them. (use this if
// you want)
#ifndef __CUDACC__

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#endif

#include <deque>
#include <vector>

using namespace std;

namespace Ped {
class Twaypoint;

class Tagent {
 public:
  Tagent(int posX, int posY);
  Tagent(double posX, double posY);

  // Returns the coordinates of the desired position
  int getDesiredX() const { return desiredPositionX; }
  int getDesiredY() const { return desiredPositionY; }

  // Sets the agent's position
  void setX(int newX) { x = newX; }
  void setY(int newY) { y = newY; }

  // Set desired position closer to destination
  void computeNextDesiredPosition();

  // Position of agent defined by x and y
  int getX() const { return x; };
  int getY() const { return y; };

  // Adds a new waypoint to reach for this agent
  void addWaypoint(Twaypoint* wp);

  int getNumWaypoints() { return waypoints.size(); }
  deque<Twaypoint*>* getWaypoints() { return &waypoints; }

 private:
  Tagent(){};

  // The agent's current position
  int x;
  int y;

  // The agent's desired next position
  int desiredPositionX;
  int desiredPositionY;

  // The current destination (may require several steps to reach)
  Twaypoint* destination;

  // The last destination
  Twaypoint* lastDestination;

  // The queue of all destinations that this agent still has to visit
  deque<Twaypoint*> waypoints;

  // Internal init function
  void init(int posX, int posY);

  // Returnst he next destination to visit
  Twaypoint* getNextDestination();
};
}  // namespace Ped

#endif
