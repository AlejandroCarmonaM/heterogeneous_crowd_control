///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
//
#include "MainWindow.h"

#include <QBrush>
#include <QGraphicsView>
#include <QImage>
#include <QtGui>
#include <algorithm>
#include <iostream>

// El modo gráfico requiere que le pasemos el vector de agentes. A no ser que
// queramos crear un vector nuevo para cada frame (que es complicado porque es
// de referencias a agentes) tenemos que cambiar el código que muestra los
// agentes por pantalla

#define ASSIGNMENT_4

MainWindow::MainWindow(const Ped::Model& model) : model(model) {
  // The Window
  graphicsView = new QGraphicsView();

  setCentralWidget(graphicsView);

  // A surface for managing a large number of 2D graphical items
  scene = new QGraphicsScene(QRect(0, 0, 800, 600), this);

  // Connect
  graphicsView->setScene(scene);

  for (int x = 0; x <= 800; x += cellsizePixel) {
    scene->addLine(x, 0, x, 800, QPen(Qt::gray));
  }

  // Now add the horizontal lines, paint them green
  for (int y = 0; y <= 800; y += cellsizePixel) {
    scene->addLine(0, y, 800, y, QPen(Qt::gray));
  }

  // Create viewAgents with references to the position of the model counterparts
  Ped::TagentSoA* agents_soa = model.getAgentSoA();
  agents_x = agents_soa->getAgentsX();
  agents_y = agents_soa->getAgentsY();
  n_agents = agents_soa->getNumAgents();
  rects = new QGraphicsRectItem*[n_agents];

  QBrush blueBrush(Qt::green);
  QPen outlinePen(Qt::black);
  outlinePen.setWidth(2);

  for (int i = 0; i < n_agents; i++) {
    rects[i] = scene->addRect(MainWindow::cellToPixel(agents_x[i]),
                              MainWindow::cellToPixel(agents_y[i]), MainWindow::cellsizePixel - 1,
                              MainWindow::cellsizePixel - 1, outlinePen, blueBrush);
  }

  if (model.getImplementation() == Ped::IMPLEMENTATION::COL_PREVENT_PAR ||
      model.getImplementation() == Ped::IMPLEMENTATION::COL_PREVENT_SEQ) {
    QPen redPen(Qt::red);
    redPen.setWidth(5);

    auto col_box = agents_soa->getColAreaLimits();

    QBrush redBrush(Qt::red);
    QBrush transparentBrush(Qt::transparent);

    // scene->addRect(MainWindow::cellToPixel(col_box.x),
    //                MainWindow::cellToPixel(col_box.y),
    //                MainWindow::cellsizePixel * col_box.w,
    //                MainWindow::cellsizePixel * col_box.h, redPen,
    //                transparentBrush);

    rect_arr[0] =
        scene->addRect(MainWindow::cellToPixel(col_box.x), MainWindow::cellToPixel(col_box.y),
                       MainWindow::cellsizePixel * (col_box.w / 2),
                       MainWindow::cellsizePixel * (col_box.h / 2), redPen, transparentBrush);
    rect_arr[1] = scene->addRect(
        MainWindow::cellToPixel(col_box.x + (col_box.w / 2)), MainWindow::cellToPixel(col_box.y),
        MainWindow::cellsizePixel * (col_box.w / 2), MainWindow::cellsizePixel * (col_box.h / 2),
        redPen, transparentBrush);
    rect_arr[2] = scene->addRect(
        MainWindow::cellToPixel(col_box.x), MainWindow::cellToPixel(col_box.y + (col_box.h / 2)),
        MainWindow::cellsizePixel * (col_box.w / 2), MainWindow::cellsizePixel * (col_box.h / 2),
        redPen, transparentBrush);
    rect_arr[3] =
        scene->addRect(MainWindow::cellToPixel(col_box.x + (col_box.w / 2)),
                       MainWindow::cellToPixel(col_box.y + (col_box.h / 2)),
                       MainWindow::cellsizePixel * (col_box.w / 2),
                       MainWindow::cellsizePixel * (col_box.h / 2), redPen, transparentBrush);
  }

#ifdef ASSIGNMENT_4
  if (model.getHeatmap() != nullptr) {
    const int heatmapSize = model.getHeatmapSize();
    QPixmap pixmapDummy = QPixmap(heatmapSize, heatmapSize);
    pixmap = scene->addPixmap(pixmapDummy);
  }
#endif

  paint();
  graphicsView->show();  // Redundant?
}

void MainWindow::paint() {
#ifdef ASSIGNMENT_4
  if (model.getHeatmap() != nullptr) {
    const int heatmapSize = model.getHeatmapSize();
    QImage image((uchar*)*model.getHeatmap(), heatmapSize, heatmapSize, heatmapSize * sizeof(int),
                 QImage::Format_ARGB32);
    pixmap->setPixmap(QPixmap::fromImage(image));
  }
#endif
  for (int i = 0; i < n_agents; i++) {
    rects[i]->setRect(MainWindow::cellToPixel(agents_x[i]), MainWindow::cellToPixel(agents_y[i]),
                      MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1);
  }

  if (model.getImplementation() == Ped::IMPLEMENTATION::COL_PREVENT_PAR ||
      model.getImplementation() == Ped::IMPLEMENTATION::COL_PREVENT_SEQ) {
    Ped::TagentSoA* agents_soa = model.getAgentSoA();
    auto col_box = agents_soa->getColAreaLimits();

    rect_arr[0]->setRect(MainWindow::cellToPixel(col_box.x), MainWindow::cellToPixel(col_box.y),
                         MainWindow::cellsizePixel * (col_box.w / 2),
                         MainWindow::cellsizePixel * (col_box.h / 2));
    rect_arr[1]->setRect(
        MainWindow::cellToPixel(col_box.x + (col_box.w / 2)), MainWindow::cellToPixel(col_box.y),
        MainWindow::cellsizePixel * (col_box.w / 2), MainWindow::cellsizePixel * (col_box.h / 2));
    rect_arr[2]->setRect(
        MainWindow::cellToPixel(col_box.x), MainWindow::cellToPixel(col_box.y + (col_box.h / 2)),
        MainWindow::cellsizePixel * (col_box.w / 2), MainWindow::cellsizePixel * (col_box.h / 2));
    rect_arr[3]->setRect(MainWindow::cellToPixel(col_box.x + (col_box.w / 2)),
                         MainWindow::cellToPixel(col_box.y + (col_box.h / 2)),
                         MainWindow::cellsizePixel * (col_box.w / 2),
                         MainWindow::cellsizePixel * (col_box.h / 2));
  }
}

int MainWindow::cellToPixel(int val) { return val * cellsizePixel; }
