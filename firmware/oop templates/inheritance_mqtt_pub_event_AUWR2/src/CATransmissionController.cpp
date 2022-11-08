#include "CATransmissionController.h"



// Class constructor
CATransmissionController::CATransmissionController(): mp_brokerName(MQTT_BROKER_NAME),mp_brokerIP(MQTT_BROKER_IP), mi_brokerPort(MQTT_BROKER_PORT), 
                                                      mp_topic(MQTT_TOPIC), mqttClientId(MQTT_CLIENT_ID), mqttClient(wifiClient)
{
  mqttClient.setServer(this->mp_brokerIP, this->mi_brokerPort); 
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


