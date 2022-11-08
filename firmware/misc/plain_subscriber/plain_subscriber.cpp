#include <SPI.h>
//#include <WiFiNINA.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "CController.h"
#include "controllerIni.h"

static WiFiClient wifiClient;
static PubSubClient mqttClient(wifiClient);

/*please define your sensitive data in the controllerIni.h */
const char ssid[] = SECRET_SSID;              // imported from controllerIni.h 
const char pass[] = SECRET_PASS;              // imported from controllerIni.h 
const char serverAddress[] = MQTT_BROKER_IP;  // imported from controllerIni.h
const char mqttClientId[] = MQTT_CLIENT_ID;   // imported from controllerIni.h
const char broker[] = MQTT_BROKER_NAME;       // imported from controllerIni.h 
int        port     = MQTT_BROKER_PORT;       // imported from controllerIni.h 
const char topic[]  = MQTT_TOPIC;             // imported from controllerIni.h 


//char* Topic;
byte* buffer;
boolean Rflag=false;
int r_len;

void callback(char* topic, byte* payload, unsigned int length) {
   //Payload=[];
   //Topic=topic;
  Rflag=true; //will use in main loop
  r_len=length; //will use in main loop
  // Serial.print("length message received in callback= ");
  // Serial.println(length);
  for (int i=0;i<length;i++) 
  {
    buffer[i]=payload[i];
    Serial.print((char)payload[i]);
  }
}


// void reconnect() {
//   // Loop until we're reconnected
//   while (!mqttClient.connected()) {
//     //mqttClient.setServer(broker, port);
//     Serial.print("Attempting MQTT connection...");
//     // Attempt to connect
//     if (mqttClient.connect(mqttClientId)) {
//       Serial.println("connected");
//       // Once connected, publish an announcement...
//       //mqttClient.publish("outTopic","hello world");
//       // ... and resubscribe
//       mqttClient.subscribe(topic);
//     } else {
//       Serial.print("failed, rc=");
//       Serial.print(mqttClient.state());
//       Serial.println(" try again in 5 seconds");
//       // Wait 5 seconds before retrying
//       delay(5000);
//     }
//   }
// }

// void connect_to_MQTT();
// void disconnect_from_MQTT();

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
  Serial.print("Attempting to connect to SSID: ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    // failed, retry
    Serial.print(".");
    delay(5000);
  }

  Serial.println("You're connected to the network");
  Serial.println();

  Serial.print("Attempting to connect to the MQTT broker: ");
  Serial.println(broker);
  
  if (mqttClient.connect(mqttClientId)) 
  {
    // connection succeeded
    Serial.println("Connected now subscribing");
    boolean r= mqttClient.subscribe(topic);
  }
  else
  {
    Serial.print("Failed. State: ");
  
    Serial.println(mqttClient.state());
  }
}

void loop()
{
    //

  while (true)
  {
    //Serial.println("publishing string");
    char *msg="test message";
    boolean rc = mqttClient.publish("test", msg);
    delay(1000);
    mqttClient.loop();
    
    if(Rflag)
    {
      // Serial.print("Message arrived in main loop[");
      // Serial.print(Topic);
      // Serial.print("] ");
      // Serial.print("message length =");
      // Serial.print(r_len);
      for (int i=0;i<r_len;i++) 
      {
        Serial.print((char)buffer[i]);
      }
      Serial.println();
      Rflag=false;
    }
  } 
  

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