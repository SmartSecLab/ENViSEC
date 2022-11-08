/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for a      *
*   timed driven subscriber, to an MQTT-broker.                     *
*                                                                   *
*   Usce case 6.1 luminance subscriber. Part of code is based on    *
*   David A. Mellis in 2006, and later modified by Tom Igoe and     *
*   Scott Fitzgerald                                                *
*   (https://docs.arduino.cc/built-in-examples/communication/Dimmer)*
*                                                                   *
*                                                                   *
*   Coded by: Andreas Lyth (2022)                                   *
********************************************************************/ 
#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "controllerIni.h"


static WiFiClient wifiClient;
static PubSubClient mqttClient(wifiClient);

/*please define your sensitive data in the controllerIni.h */
const char* ssid = SECRET_SSID;              // imported from controllerIni.h 
const char* pass = SECRET_PASS;              // imported from controllerIni.h 
const char* serverAddress = MQTT_BROKER_IP;  // imported from controllerIni.h
const char* mqttClientId = MQTT_CLIENT_ID;   // imported from controllerIni.h
const char* broker = MQTT_BROKER_NAME;       // imported from controllerIni.h 
int         port     = MQTT_BROKER_PORT;       // imported from controllerIni.h 
const char* topic  = MQTT_TOPIC;             // imported from controllerIni.h 
int sleepCntr = 0;

const int ledPin = 9;      // the pin that the LED is attached to

void callback(char* topic, byte* payload, unsigned int length)
{
  String tempData = "";
  for (int i=0;i<length;i++) 
  {
    String s_tempChar = "";
    s_tempChar = String((char)payload[i]);
    tempData.concat(s_tempChar);
  }
  if(tempData.toInt() < 200)
  {
    analogWrite(ledPin,30);
  }
  else if((tempData.toInt() >= 200 && tempData.toInt() < 650))
  {
    analogWrite(ledPin,130);
  }
  else analogWrite(ledPin,255);

}



void setup()
{

  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  //mqttClient.setServer(serverAddress, 1883);
  mqttClient.setServer(MQTT_BROKER_IP, port);
  mqttClient.setCallback(callback);

  // attempt to connect to Wifi network:
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {

    delay(5000);
  }
  
  if (mqttClient.connect(mqttClientId)) 
  {
    boolean r= mqttClient.subscribe(topic);
  }
  else
  {
  
  }
}

void loop()
{
    mqttClient.loop();

} 