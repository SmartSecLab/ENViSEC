#ifndef CATRANSMISSIONCONTROLLER_H
#define CATRANSMISSIONCONTROLLER_H

#include <Arduino.h>
//#include <ArduinoMqttClient.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include "controllerIni.h"

// extern IPAddress server();
// extern WiFiClient wifiClient;
// extern MqttClient mqttClient;

class CATransmissionController
{
private:
  const char* mp_brokerName ;
  const char* mp_brokerIP;
  const int mi_brokerPort;
  const char* mp_topic;
  const char* mqttClientId; 
  //const char *ssid = SECRET_SSID;            // your network SSID (name)
  //const char *pass = SECRET_PASS;            // your network password (use for WPA, or use as key for WEP)
  //IPAddress serverAddress;
  WiFiClient wifiClient;
  PubSubClient mqttClient;

protected:
  CATransmissionController();
  virtual bool m_virtualTransmission() = 0;
  int mi_data;
  void m_connect_to_MQTT();
  void m_disconnect_from_MQTT();
  void m_transmitData();
  void m_connect_to_WiFi();

public:
};

#endif