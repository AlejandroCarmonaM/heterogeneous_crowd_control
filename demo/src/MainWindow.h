///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
#ifndef _mainwindow_h_
#define _mainwindow_h_

#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QLabel>
#include <QMainWindow>
#include <vector>

#include "ViewAgent.h"
#include "ped_agent.h"
#include "ped_model.h"

////////////
/// Comment in for Assignment 4
///////////////////////////////////////////////
// #define ASSIGNMENT_4

class QGraphicsView;

class MainWindow : public QMainWindow {
 public:
  MainWindow() = delete;
  MainWindow(const Ped::Model& model);
  ~MainWindow() {
    if (rects != nullptr) {
      delete[] rects;
      rects = nullptr;
    }
  }

  // paint is called after each computational step
  // to repaint the window
  void paint();

  static int cellToPixel(int val);
  static const int cellsizePixel = 5;

 private:
  QGraphicsView* graphicsView;
  QGraphicsScene* scene;

  const Ped::Model& model;

  // the graphical representation of each agent
  std::vector<ViewAgent*> viewAgents;

  // The pixelmap containing the heatmap image (Assignment 4)
  QGraphicsPixmapItem* pixmap;

  int n_agents;
  int* agents_x;
  int* agents_y;
  QGraphicsRectItem** rects = nullptr;
  QGraphicsRectItem* rect_arr[4];
};

#endif
