#if !defined(CASLEEPCONTROLLER_H)
#define CASLEEPCONTROLLER_H

#include <avr/sleep.h>
#include <avr/interrupt.h>
#include "controllerIni.h"

#define SECS_IN_ONE_MIN_CONSTANT 6   //Used to calculate sleep interval in minutes. Do not change

class CASleepController
{
private:
  int mi_sleepCntr;
  bool mb_status = false;
  void m_incrementSleepCounter();
  void m_resetSleepCounter();
  
protected:
  CASleepController();
  virtual bool m_virtualSleep() = 0;
  bool m_checkForWakeup();
  void m_deviceSleep();
  
public:
};

#endif