/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for a      *
*   timed driven subscriber, to an MQTT-broker.                     *
*                                                                   *
*                                                                   *
*   Class system overview:                                          *
*  -AbstractSleepController:                                        *
*     Abstract class for controlling device sleep. Contains macros  *
*     for user to control messaging interval and member functions   *
*     for transmission of messages to broker                        *
*  -AbstractTransmissionController:                                 *
*     Abstract class containing member functions for sending        *
*     messages to MQTT broker.                                      *
*  -Controller:                                                     *
*     Instantiable class for controlling device sleep and data      *
*     transmission. Class inherits from abstract classes            *
*     AbstractSleepController and AbstractTransmissionController    *
*     These classes work as function containers for their           *
*     respective area, called through the the derived Controller    *
*     class                                                         *
*                                                                   *
*                                                                   *
*   Coded by: Andreas Lyth (2022)                                   *
********************************************************************/ 
#include <SPI.h>
//#include <WiFiNINA.h>
#include <WiFi.h>
#include <PubSubClient.h>
// #include "CController.h"
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


void callback(char* topic, byte* payload, unsigned int length) {
  for (int i=0;i<length;i++) 
  {
    Serial.print((char)payload[i]);
  }
  Serial.println("");
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








////////////////////////////////////////////////////////////////////////
// #include <Arduino.h>
// #include "CController.h"
// //
// // #include <avr/sleep.h>
// // #include <avr/interrupt.h>
// // #include <PubSubClient.h>
// // #include <WiFi.h>
// // #include "controllerIni.h"

// const char* const ssid = SECRET_SSID;            // your network SSID (name)
// const char* const pass = SECRET_PASS;            // your network password (use for WPA, or use as key for WEP)

// // static WiFiClient g_wifiClient;
// // static PubSubClient g_mqttClient(g_wifiClient);
// CController& controller = CController::instance();

// // void mainFunctionallity();
// // void globalCallback(char* topic, byte* payload, unsigned int length) {
// //   for (int i=0;i<length;i++) 
// //   {
// //     Serial.print((char)payload[i]);
// //   }
// // }

// int sensorData = 0;


// void setup()
// {
//   //Initialize serial and wait for port to open:
//   Serial.begin(9600);
//   while (!Serial) {
//     ; // wait for serial port to connect. Needed for native USB port only
//   }
  
//   // g_mqttClient.setServer(MQTT_BROKER_IP, MQTT_BROKER_PORT);
//   // g_mqttClient.setCallback(globalCallback);

//   //attempt to connect to Wifi network:
//   while (WiFi.begin(ssid, pass) != WL_CONNECTED)
//   {
//     delay(5000);
//   }
//   Serial.println("Wifi");
  
//   // if (g_mqttClient.connect(MQTT_CLIENT_ID)) 
//   // {
//   //   // connection succeeded
//   //   Serial.println("Connected now subscribing");
//   //   boolean r= g_mqttClient.subscribe(MQTT_TOPIC,1);
//   // }
//   // else
//   // {
//   //   Serial.print("Failed. State: ");
  
//   //   Serial.println(g_mqttClient.state());
//   // }
   

// }



// void loop()
// {
//   /* DO NOT WRITE ANY CODE HERE: INSERT YOUR CODE 
//   IN THE FUNCTION BELOW, NAMED "mainFunctionallity()" */
//   controller.m_processController();
// Serial.println(sensorData);
// delay(200);
// sensorData++;
  
//   //g_mqttClient.loop();
//   //controller.mqttClient.loop();
// }

// void mainFunctionallity()
// {
  
// }

