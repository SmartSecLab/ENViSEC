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

/* use case specific declarations*/



void callback(char* topic, byte* payload, unsigned int length)
{
  String tempData = "";
  for (int i=0;i<length;i++) 
  {
    String s_tempChar = "";
    s_tempChar = String((char)payload[i]);
    tempData.concat(s_tempChar);
  }
  if(tempData.toInt() < 15)
  {
    PORTE_OUTSET = PIN3_bm;
  }
  else
  {
    PORTE_OUTCLR = PIN3_bm;

  }

}

void setup()
{
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  /* use case specific setup: Red led for indicating
    bin is ready to be emptied*/
  PORTE_PIN3CTRL |= PORT_PULLUPEN_bm;
  PORTE_DIRSET |= PIN3_bm;

  //mqttClient.setServer(serverAddress, 1883);
  mqttClient.setServer(MQTT_BROKER_IP, port);
  mqttClient.setCallback(callback);

  // attempt to connect to Wifi network:
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {

    delay(5000);
  }
  
  if (mqttClient.connect(mqttClientId)) 
  {
    boolean r= mqttClient.subscribe(topic,1);
  }
}

void loop()
{
  mqttClient.loop();
} 
