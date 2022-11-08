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
  CController();
protected:

public:
  CController(const CController&) = delete;
  CController operator=(const CController&) = delete;
  void m_setData(int mi_data);
  int m_getData();
  void m_processController();
  static CController& instance(){
    static CController instance;
    return instance;
  };
};

//extern CController controllerObj;
#endif // DEVICE_H
