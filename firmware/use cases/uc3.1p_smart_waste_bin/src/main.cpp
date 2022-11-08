/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for a      *
*   timed driven subscriber, to an MQTT-broker.                     *
*                                                                   *
*   Use case 3:                                                     * 
*   Smart Waste Bin. Code loosely base on project project available *
*    at (https://create.arduino.cc/projecthub/abdularbi17/          *
*        ultrasonic-sensor-hc-sr04-with-arduino-tutorial-327ff6)    *
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

/* use case specific declarations*/
#define echoPin 2 // attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin 3 //attach pin D3 Arduino to pin Trig of HC-SR04
long duration; // variable for the duration of sound wave travel
int distance; // variable for the distance measurement


void setup()
{
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  /* use case specific setup */
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin, INPUT); // Sets the echoPin as an INPUT

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
  // Clears the trigPin condition
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2; // Speed of sound wave divided by 2 (go and back)
  // Displays the distance on the Serial Monitor
  controller.m_setData(distance);
}




