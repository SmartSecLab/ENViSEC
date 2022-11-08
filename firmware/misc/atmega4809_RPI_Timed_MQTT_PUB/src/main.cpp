#include <Arduino.h>
#include <ArduinoMqttClient.h>
#include <WiFiNINA.h>
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <avr/io.h>
#include "connection_control.h"
#include "disable_ports.h"


#define SLEEP_INTERVAL_MINUTES 10     //sleep interval in minutes
#define SECS_IN_ONE_MIN_CONSTANT 60   //Used to calculate sleep interval in minutes. Do not change

/* please enter your sensitive data in connection_control.h */
char ssid[] = SECRET_SSID;            // your network SSID (name)
char pass[] = SECRET_PASS;            // your network password (use for WPA, or use as key for WEP)


static WiFiClient wifiClient;
static MqttClient mqttClient(wifiClient);


IPAddress server(192,168,0,100); 
const char broker[] = "envisecmqtt1";
int        port     = 1883;
const char topic1[]  = "UnoTestTopic";


/* Function declarations */
void connect_to_MQTT();
void disconnect_from_MQTT();
void transmitData(int sensorData);
void deviceSleep();

/* Global variables */
static int sleepCntr = 0;
int testCntr = 0; //debug


void setup()
{
  /* debug: leds */
   PORTD_PIN6CTRL |= PORT_PULLUPEN_bm;
   PORTD_DIRSET |= PIN6_bm;  //Set PD6 to output

  /* Sleep mode setup */
  set_sleep_mode(SLPCTRL_SMODE_PDOWN_gc);

  /* Wake up timer setup (PIT) */
  RTC_PITCTRLA = RTC_PERIOD_CYC32768_gc    //Set period length for periodic interrupt (1 sec)
                  | RTC_RTCEN_bm;           //Enable RTC peridoc interrupt
  RTC_PITINTCTRL = RTC_PI_bm;              //PIT interrupt enable

      while (WiFi.begin(ssid, pass) != WL_CONNECTED)
    {
       delay(5000);
    }

}//end setup

ISR(RTC_PIT_vect)
{
  /* debug: toggle onboard led */
  RTC_PITINTFLAGS |= RTC_PITEN_bm;
  PORTD_OUTTGL = PIN6_bm;


}

void loop()
{
  /* Test: Application run time */
  if(sleepCntr >= (SLEEP_INTERVAL_MINUTES * SECS_IN_ONE_MIN_CONSTANT) )
  {
    /* Debug: increment testcounter for message tracking */
    testCntr++;

    /* Connect to WLAN and MQTT broker */
    connect_to_MQTT();

    /* Device Main Functionality */
    /*                           */
    /*        Insert the         */
    /*           Main            */
    /*      Functionallity       */
    /*       of the device       */
    /*           here            */
    /*___________________________*/

    /* Device application and data transmit */
     transmitData(testCntr); 

    /* disconnect from MQTT broker and WLAN*/
     disconnect_from_MQTT();

    /* Reset sleep counter */
    sleepCntr = 0;
  }

  /* Put device to sleep */
  deviceSleep();

}//end main loop


/* Function for connectin device to
   WLAN and MQTT broker */
void connect_to_MQTT()
{
  /* check WLAN connection */
  if(WiFi.status() != WL_CONNECTED)
  {
    while (WiFi.begin(ssid, pass) != WL_CONNECTED)
    {
       delay(5000);
    }
  }

  if(!mqttClient.connected())
  {
    mqttClient.connect(server, port);
  }
}


/* Function for disconnecting device 
   from MQTT broker and WLAN */
void disconnect_from_MQTT()
{
  /* disconnect from MQTT broker */
  if(mqttClient.connected())
  {
    mqttClient.stop();
  }
}


/* Function for transmitting
   sensor data to MQTT broker */
void transmitData(int sensorData)
{
  mqttClient.beginMessage(topic1, false, 1, false);  //params: const char* topic, bool retain, uint8_t qos, bool dup
  mqttClient.print(sensorData);
  mqttClient.endMessage();
}


/* Function for putting 
   device to sleep */
void deviceSleep()
{
  sleep_enable();
  sei();
  sleep_cpu();

  /* increment sleep counter 
     for application run time */
  sleepCntr++;

  sleep_disable();
  cli();

}

