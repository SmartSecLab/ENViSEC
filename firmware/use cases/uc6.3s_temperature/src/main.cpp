/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for a      *
*   timed driven subscriber, to an MQTT-broker.                     *
*                                                                   *
*   Usce case 6.1 luminance subscriber. Part of code is based on    *
*   a tutorial by Random Nerd                                       *
*   (https://randomnerdtutorials.com/esp32-mqtt-publish-            *
*   subscribe-arduino-ide/)*                                        *
*                                                                   *
*                                                                   *
*   Coded by: Andreas Lyth (2022)                                   *
********************************************************************/ 
#include <Arduino.h>
#include <Adafruit_Sensor.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <WiFi.h>
#include "controllerIni.h"


// Replace the next variables with your SSID/Password combination
const char* ssid  = SECRET_SSID;  
const char* password  = SECRET_PASS; 
const char* mqtt_server = MQTT_BROKER_IP;
const unsigned long pollingInterval = 10000;
unsigned long PollingTime = 0;

/* use case specific declarations */
WiFiClient espClient;
PubSubClient client(espClient);
double temperature =0.0;
double humidity = 0.0;
char tempStr[8];
char humiStr[8];
const int redLed = 17; //17
const int blueLed = 21;//21
void callback(char* topic, byte* message, unsigned int length);


void setup() {
  pinMode(redLed, OUTPUT);
  pinMode(blueLed, OUTPUT);

  WiFi.begin(SECRET_SSID, SECRET_PASS);
  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(500);
  }
  client.setServer(MQTT_BROKER_IP, MQTT_BROKER_PORT);
  client.setCallback(callback);
}


void callback(char* topic, byte* message, unsigned int length) 
{
  String tempData = "";
  for (int i=0;i<length;i++) 
  {
    String s_tempChar = "";
    s_tempChar = String((char)message[i]);
    tempData.concat(s_tempChar);
  }

  if(tempData.toDouble() >=UPPER_THRESHOLD)
  {
    digitalWrite(redLed, LOW);
    digitalWrite(blueLed,HIGH);
  }
  else if(tempData.toDouble() < UPPER_THRESHOLD)
  {
    digitalWrite(blueLed,LOW);
    digitalWrite(redLed, HIGH);
  }
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    // Attempt to connect
    if (client.connect(MQTT_CLIENT_ID))
    {
      client.subscribe(MQTT_TOPIC1);
    } 
    else 
    {
      delay(5000);
    }
  }
}

//main loop of program
void loop() {
 if(millis() >= PollingTime)
  {
    PollingTime += pollingInterval;
    if (!client.connected())
    {
     reconnect();
    }
    client.loop();
  }
}