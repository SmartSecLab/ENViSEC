#ifndef CCONTROLLER_H
#define CCONTROLLER_H

//#include <Arduino.h>
#include "CASleepController.h"
#include "CATransmissionController.h"
//#include "controllerConfig.h"

extern void mainFunctionallity();     //scope: Global (main)

class CController: protected CASleepController, protected CATransmissionController
{
private:
  void m_disablePorts();
  friend void mainFunctionallity();   //scope: Global (main)
  bool m_virtualSleep();
  bool m_virtualTransmission();
  
protected:

public:
  CController();
  void m_setData(int mi_data);
  int m_getData();
  void m_processController();
};

extern CController controllerObj;
#endif // DEVICE_H
