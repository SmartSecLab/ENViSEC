#ifndef CATRANSMISSIONCONTROLLER_H
#define CATRANSMISSIONCONTROLLER_H

#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include "controllerIni.h"


class CATransmissionController
{
private:
  const char* mp_brokerName ;
  const char* mp_brokerIP;
  const int mi_brokerPort;
  const char* mp_topic;
  const char* mqttClientId; 
  WiFiClient wifiClient;
  PubSubClient mqttClient;
  /////////////
  void m_callBack(char* topic, byte* payload, unsigned int length);
  void (*callBackPtr)(char* topic, byte* payload, unsigned int length);
  ////
  byte* buffer;
  bool Rflag=false;
  int r_len;

protected:
  CATransmissionController();
  virtual bool m_virtualTransmission() = 0;
  int mi_data;
  void m_connect_to_MQTT();
  void m_disconnect_from_MQTT();
  void m_transmitData();
  void m_connect_to_WiFi();
  //virtual void onCallback(char* topic, byte* payload, unsigned int len);
  void m_incomingMsg();
};

#endif