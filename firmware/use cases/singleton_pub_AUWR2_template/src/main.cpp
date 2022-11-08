/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for a      *
*   timed driven subscriber, to an MQTT-broker.                     *
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


void setup()
{
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("connected to wifi"); 
  // attempt to connect to Wifi network:
  while (WiFi.begin(ssid, pass) != WL_CONNECTED)
  {
    delay(5000);
  }
  Serial.println("connected to mqtt"); 
}

void loop()
{
  /* DO NOT WRITE ANY CODE HERE: INSERT YOUR CODE 
  IN THE FUNCTION BELOW, NAMED "mainFunctionallity()" */
  controller.m_processController();
  
}

void mainFunctionallity()
{
  sensorData++;
  controller.m_setData(sensorData);
  Serial.println(sensorData);
}

