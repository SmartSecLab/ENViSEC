#include "CATransmissionController.h"

// Class constructor
CATransmissionController::CATransmissionController(): mp_brokerName(MQTT_BROKER_NAME),mp_brokerIP(MQTT_BROKER_IP), mi_brokerPort(MQTT_BROKER_PORT), 
                                                      mp_topic(MQTT_TOPIC), mqttClientId(MQTT_CLIENT_ID), mqttClient(wifiClient)
{
  //serverAddress.fromString(MQTT_BROKER_IP);
  mqttClient.setServer(this->mp_brokerIP, this->mi_brokerPort);
  //callBackPtr = &m_callBack;
  //mqttClient.setCallback(m_callBack);
  mqttClient.setCallback([](char* topic,uint8_t* payload,unsigned int length){
    CControler::instance.m_callback
  });
  //mqttClient.setCallback([this] (char* topic, byte* payload, unsigned int length) 
  //  { this->m_callback(topic, payload, length); });
}


/**************************************************
*                MEMBER FUNCTIONS                 *
**************************************************/

/* Function for connectin device to
   WLAN and MQTT broker */
void CATransmissionController::m_connect_to_MQTT()
{
  /* check MQTT broker connection */
  if(!mqttClient.connected())
  {
    mqttClient.connect(mqttClientId);
  }
}

/* Function for disconnecting device 
   from MQTT broker and WLAN */
void CATransmissionController::m_disconnect_from_MQTT()
{
  /* disconnect from MQTT broker */
  if(mqttClient.connected())
  {
    mqttClient.disconnect();
  }
}

/* Function for transmitting
   sensor data to MQTT broker */
void CATransmissionController::m_transmitData()
{
  char payload[33];
  itoa(mi_data,payload,10);
  mqttClient.publish(mp_topic, payload);
}

void CATransmissionController::m_callBack(char* topic, byte* payload, unsigned int length) 
{
  this->Rflag = true; 
  this->r_len = length; 
  for (int i=0; i<length;i++) 
  {
    this->buffer[i]=payload[i];
    Serial.print((char)payload[i]);
  }
}

void CATransmissionController::m_incomingMsg()
{
  while (true)
  {
    //Serial.println("publishing string");
    char *msg="test message";
    boolean rc = mqttClient.publish("test", msg);
    delay(1000);
    mqttClient.loop();
    
    if(Rflag)
    {
      for (int i=0;i<r_len;i++) 
      {
        Serial.print((char)buffer[i]);
      }
      Serial.println();
      Rflag=false;
    }
  } 
}

