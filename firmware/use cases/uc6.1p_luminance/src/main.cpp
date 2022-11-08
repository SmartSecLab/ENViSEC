

/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for a      *
*   timed driven subscriber, to an MQTT-broker.                     *
*                                                                   *
*   Use case 6.1 luminance meter                                                                *
*                                                                   *
*                                                                   *
*                                                                   *
*   Coded by: Andreas Lyth (2022)                                   *
********************************************************************/ 
#include <Arduino.h>
#include "CController.h"

const char* const ssid = SECRET_SSID;            // your network SSID (name)
const char* const pass = SECRET_PASS;            // your network password (use for WPA, or use as key for WEP)

void mainFunctionallity();

CController& controller = CController::instance();
int sensorData = 0;

/* use case specific definitions and variables. */
int analogPin = A0;

void setup()
{  
  // attempt to connect to Wifi network:
  while (WiFi.begin(ssid, pass) != WL_CONNECTED)
  {
    delay(5000);
  }
}

void loop()
{
  /* DO NOT WRITE ANY CODE HERE: INSERT YOUR CODE 
  IN THE FUNCTION BELOW, NAMED "mainFunctionallity()" */
  controller.m_processController();
}

void mainFunctionallity()
{
  sensorData = analogRead(analogPin);
  controller.m_setData(sensorData);
}

