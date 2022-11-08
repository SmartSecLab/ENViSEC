#include "CATransmissionController.h"

// Class constructor
CATransmissionController::CATransmissionController(): mp_brokerName(MQTT_BROKER_NAME),mp_brokerIP(MQTT_BROKER_IP), mi_brokerPort(MQTT_BROKER_PORT), 
                                                      mp_topic(MQTT_TOPIC), mqttClientId(MQTT_CLIENT_ID), mqttClient(wifiClient)
{
  //serverAddress.fromString(MQTT_BROKER_IP);
  mqttClient.setServer(this->mp_brokerIP, this->mi_brokerPort); 
}


/**************************************************
*                MEMBER FUNCTIONS                 *
**************************************************/

/* Function for connectin device to
   WLAN and MQTT broker */
void CATransmissionController::m_connect_to_MQTT()
{
  /* check WLAN connection */
  if(WiFi.status() != WL_CONNECTED)
  {
    // while (WiFi.begin(ssid, pass) != WL_CONNECTED)
    // {
    //    delay(5000);
    // }
  }

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
  // mqttClient.beginMessage(mp_topic, false, 1, false);  //params: const char* topic, bool retain, uint8_t qos, bool dup
  // mqttClient.print(this->mi_data);
  // mqttClient.endMessage();
  char payload[33];
  itoa(mi_data,payload,10);
  mqttClient.publish(mp_topic, payload);
}


