#ifndef CCONTROLLER_H
#define CCONTROLLER_H

//#include <Arduino.h>
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include "controllerIni.h"

extern void mainFunctionallity();     //scope: Global (main)

class CController
{
private:
/*CController*/
  CController();
  void m_disablePorts();
  friend void mainFunctionallity();   //scope: Global (main)
  bool m_virtualSleep();
  bool m_virtualTransmission();
/*Sleep*/
  int mi_sleepCntr;
  bool mb_status = false;
  void m_incrementSleepCounter();
  void m_resetSleepCounter();
/*Transmission*/
  WiFiClient wifiClient;
  PubSubClient mqttClient;
  bool m_ThresholdCheck();
  void m_callBack(char* topic, byte* payload, unsigned int length);

/*Sleep*/
  bool m_checkForWakeup();
  void m_deviceSleep();
/*Transmission*/
  int mi_pubData;
  int mi_subData;
  void m_connect_to_MQTT();
  void m_disconnect_from_MQTT();
  void m_transmitData();
  void m_connect_to_WiFi();

public:
  CController(const CController&) = delete;
  CController operator=(const CController&) = delete;
  void m_setData(int mi_data);
  int m_getData();
  void m_processController();
  static CController& instance()
  {
    static CController instance;
    return instance;
  }
/*Sleep*/
/*Transmission*/

};

#endif

