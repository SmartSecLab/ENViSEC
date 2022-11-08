#include <Arduino.h>
#include <Adafruit_Sensor.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <Wire.h>
#include <WiFi.h>
#include "controllerIni.h"

#define DHTPIN 21   //DHT11 INPUT PIN ON ESP32
#define DHTTYPE DHT11   // DHT 11 TEMPERATURE AND HUMIDITY SENSOR
#define POLLING_INTERVAL 10000UL  //SENSOR POLLING INTERVAL IN MS

// Replace the next variables with your SSID/Password combination
// const char* ssid  = SECRET_SSID;  
// const char* password  = SECRET_PASS; 
// const char* mqtt_server = MQTT_BROKER_IP;

/* use case specific declarations */
WiFiClient espClient;
PubSubClient client(espClient);
DHT dht = DHT(DHTPIN, DHTTYPE);
unsigned long pollingTime = POLLING_INTERVAL;

double temperature =0.0;
double humidity = 0.0;
char tempStr[8];
char humiStr[8];

void setup() {
  pinMode(BUILTIN_LED, OUTPUT);
  dht.begin();

  WiFi.begin(SECRET_SSID, SECRET_PASS);

  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(500);
  }
  client.setServer(MQTT_BROKER_IP, MQTT_BROKER_PORT);
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    // Attempt to connect
    if (client.connect(MQTT_CLIENT_ID)) {
    } else {
      delay(5000);
    }
  }
}

//main loop of program
void loop() {
  //check for sensor polling time
  //prints sensor data to terminal 
  if(millis() >= pollingTime)
  {
    pollingTime += POLLING_INTERVAL;
    if (!client.connected()) reconnect();
    
    temperature = dht.readTemperature();
    dtostrf(temperature, 1, 2, tempStr);
    client.publish(MQTT_TOPIC1, tempStr);

    humidity = dht.readHumidity();
    dtostrf(humidity, 1, 2, humiStr);
    client.publish(MQTT_TOPIC2, humiStr);
  }

  //run other code
}