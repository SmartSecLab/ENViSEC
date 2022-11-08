/********************************************************************
*   This experimental code is written for the ENViSEC-Project at    *
*   Kristiania University College in Oslo (2022). The code is not   *
*   intended for commercial use. This code is a template for time   *
*   driven publisher, to an MQTT-broker.                            *
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
*   coded by: Andreas Lyth (2022)                                   *
********************************************************************/ 

#include <Arduino.h>
#include "Controller.h"

const char* const ssid = SECRET_SSID;            // your network SSID (name)
const char* const pass = SECRET_PASS;            // your network password (use for WPA, or use as key for WEP)

/* Declarations of global vars and functions */
void mainFunctionallity();
int sensorData = 0;

void setup()
{ 
  /* Setup your device here */
  pinMode(A1, INPUT);


  /* Setup and connecting to WiFi based on 
  information in connectionControl.h*/
  // while (WiFi.begin(ssid, pass) != WL_CONNECTED)
  // {
  //   delay(5000);
  // }
}//end setup


void loop()
{
  /* DO NOT WRITE ANY CODE HERE: INSERT YOUR CODE 
  IN THE FUNCTION BELOW, NAMED "mainFunctionallity()" */
  controllerObj.m_processController();
}//end main loop


/*Function for inserting main functionallity of the 
  device. This function is called and executed
  in controllerObj-class at the right time . 
  NOTE: data to be transmitted to the MQTT-broker
  must be stored int the global var named "data" (int) */
void mainFunctionallity()
{
  /***********************************************
  *                                              *
  *    Insert your code in this function.        *
  *    Store the data to be transmitted to       *
  *    the MQTT-broker in the variable named     *
  *    "sensorData" (int). mainFunctionallity    *
  *    must end with call to                     *
  *    controllerObj.m_setData(sensorData)       *
  *                                              *
  ***********************************************/

  sensorData = analogRead(A1); //debug
  
  /* Member function call in instanciated objective
    for setting data to be sent to MQTT-Broker */
  controllerObj.m_setData(sensorData);
}