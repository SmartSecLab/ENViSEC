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
#include <Arduino.h>
#include "CController.h"

const char* const ssid = SECRET_SSID;            // your network SSID (name)
const char* const pass = SECRET_PASS;            // your network password (use for WPA, or use as key for WEP)

void mainFunctionallity();
int sensorData = 0;


void callback(char* topic, byte* payload, unsigned int length) 
{
  Rflag=true; //will use in main loop
  r_len=length; //will use in main loop
  for (int i=0;i<length;i++) 
  {
    buffer[i]=payload[i];
    Serial.print((char)payload[i]);
  }
}

void setup()
{

  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  CController::instance()
  //mqttClient.setServer(serverAddress, 1883);
  //mqttClient.setCallback(callback);

  // attempt to connect to Wifi network:
  while (WiFi.begin(ssid, pass) != WL_CONNECTED)
  {
    delay(5000);
  }

  // Serial.println("You're connected to the network");
  // Serial.println();

  // Serial.print("Attempting to connect to the MQTT broker: ");
  // Serial.println(broker);
  
  // if (mqttClient.connect(mqttClientId)) 
  // {
  //   // connection succeeded
  //   Serial.println("Connected now subscribing");
  //   boolean r= mqttClient.subscribe(topic);
  // }
  // else
  // {
  //   Serial.print("Failed. State: ");
  
  //   Serial.println(mqttClient.state());
  // }
}

void loop()
{
  /* DO NOT WRITE ANY CODE HERE: INSERT YOUR CODE 
  IN THE FUNCTION BELOW, NAMED "mainFunctionallity()" */
  controllerObj.m_processController();


  

//   // call poll() regularly to allow the library to receive MQTT messages and
//   // send MQTT keep alive which avoids being disconnected by the broker
// Serial.print("debug");
// delay(500);
//   connect_to_MQTT();
//     delay(1000);
//   if(mqttClient.connected()){

  
//   Serial.print("Subscribing to topic: ");
//   Serial.println(topic);
//   Serial.println();

//   // subscribe to a topic
//   mqttClient.subscribe(topic);


//   // topics can be unsubscribed using:
//   // mqttClient.unsubscribe(topic);

//   Serial.print("Topic: ");
//   Serial.println(topic);

//   Serial.println();

//   mqttClient.poll();
  

//   Serial.println("Checking topic:");
//   if(mqttClient.available()) 
//   {
//     Serial.println("Message availible:");
//     Serial.print("Message: ");
//     Serial.println((char)mqttClient.read());
//   }
//   else Serial.println("Message not availible:");
//   }
//   delay(2000);
//   disconnect_from_MQTT();
//   delay(3000);
}



// void connect_to_MQTT()
// {
//   /* check WLAN connection */
//   // if(WiFi.status() != WL_CONNECTED)
//   // {
//   //   while (WiFi.begin(ssid, pass) != WL_CONNECTED)
//   //   {
//   //      delay(5000);
//   //   }
//   // }

//   while(!mqttClient.connected())
//   {
//     // mqttClient.connect(*broker, port);
//     mqttClient.connect(serverAddress, port);
//     Serial.print(".");
//     delay(2000);
//   }
//   Serial.println("CONNECTEDCONNECTEDCONNECTED");
// }

// void disconnect_from_MQTT()
// {
//   /* disconnect from MQTT broker */
//   if(mqttClient.connected())
//   {
//     mqttClient.stop();
//     Serial.println("Disconnected");
//   }
// }
