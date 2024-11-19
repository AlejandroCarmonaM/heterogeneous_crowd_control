///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
//
#include "ViewAgent.h"

#include <QGraphicsItemAnimation>

#include "MainWindow.h"

ViewAgent::ViewAgent(Ped::Tagent* agent, QGraphicsScene* scene) : agent(agent) {
  QBrush blueBrush(Qt::green);
  QPen outlinePen(Qt::black);
  outlinePen.setWidth(2);

  rect = scene->addRect(MainWindow::cellToPixel(agent->getX()),
                        MainWindow::cellToPixel(agent->getY()), MainWindow::cellsizePixel - 1,
                        MainWindow::cellsizePixel - 1, outlinePen, blueBrush);
}

void ViewAgent::paint() {
  rect->setRect(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()),
                MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1);
}
