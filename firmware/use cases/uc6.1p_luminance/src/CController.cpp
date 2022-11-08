#include "CController.h"

/* Constructor */
CController::CController(): mqttClient(wifiClient) 
                              
{

/*Sleep*/
  set_sleep_mode(SLPCTRL_SMODE_PDOWN_gc);
  /* Wake up timer setup (PIT) */
  RTC_PITCTRLA = RTC_PERIOD_CYC32768_gc    //Set period length for periodic interrupt (1 sec)
                  | RTC_RTCEN_bm;           //Enable RTC peridoc interrupt
  RTC_PITINTCTRL = RTC_PI_bm;              //PIT interrupt enable

/*Transmission*/
  mqttClient.setServer(MQTT_BROKER_IP, MQTT_BROKER_PORT);

};


/**************************************************
*         CCONTROLLER MEMBER FUNCTIONS            *
**************************************************/

/* Function for setting data member in abstract 
  parent class Virtual_Transmission_Controller */
void CController::m_setData(int data) {this->mi_pubData = data;/*Serial.println(this->mi_pubData);*/}

/* Function for getting data member in abstract 
  parent class Virtual_Transmission_Controller */
int CController::m_getData() {return this->mi_subData;}


void CController::m_processController()
{
  if(this->m_checkForWakeup())          //scope: AbstractSleepController 
  {
    mainFunctionallity();               //scope: Global
    this->m_connect_to_MQTT();          //scope: AbstractTransmissionController
    this->m_transmitData();             //scope: AbstractTransmissionController 
    this->m_disconnect_from_MQTT();     //scope: AbstractTransmissionController
    this->m_deviceSleep();              //scope: AbstractSleepController
  }
  this->m_deviceSleep();
}


/* Function for disabling ports on devicet 
  parent class Virtual_Transmission_Controller */
void CController::m_disablePorts(void){
  for (volatile uint8_t* p = &PORTA.PIN0CTRL; p <= &PORTF.PIN0CTRL; p +=0x20)
  {
    for(uint8_t i = 0; i < 8; i++) p[i] = PORT_ISC_INPUT_DISABLE_gc; //pin dinput disable
  }
}


/**************************************************
*              SLEEP FUNCTIONALITY                *
**************************************************/

  /* Function for checking for device wake up, 
   wakeing up involves commencing device's 
   Main Functionality and transmission of data
   to broker */
bool CController::m_checkForWakeup()
{
  if(this->mi_sleepCntr >= (SLEEP_INTERVAL_MINUTES * SECS_IN_ONE_MIN_CONSTANT) ) 
  {
    this->m_resetSleepCounter();
    return true;
  } 
  else return false; 
}


/* Function for putting 
   device to sleep */
void CController::m_deviceSleep()
{
  sleep_enable();                 //avr/sleep.h
  sei();                          //avr/interrupt.h
  sleep_cpu();                    //avr/sleep.h

  /* increment sleep counter 
     for application run time */
  this->m_incrementSleepCounter();
  
  sleep_disable();                //avr/sleep.h
  cli();                          //avr/interrupt.h
}


/* Function for Incrementing sleep counter,
   used for  checking for wake up interval */
void CController::m_incrementSleepCounter()
{
  this->mi_sleepCntr++;
}


/* Function for resetting sleep counter after 
   transmission of data to MQTT broker*/
void CController::m_resetSleepCounter()
{
  this->mi_sleepCntr = 0;
}


/* Function for allowing external interrupts
  to make the PIT wake up the device from sleep*/
ISR(RTC_PIT_vect)
{
  RTC_PITINTFLAGS |= RTC_PITEN_bm;
}

/**************************************************
*           TRANSMISSION FUNCTIONALITY            *
**************************************************/

/* Threshold check for event based publishing */
bool CController::m_ThresholdCheck()
{
  if(this->mi_pubData > UPPER_THRESHOLD || this->mi_pubData < LOWER_THRESHOLD) return true;
  else return false;
}


/* Function for connectin device to
   WLAN and MQTT broker */
void CController::m_connect_to_MQTT()
{
  /* check MQTT broker connection */
  if(!mqttClient.connected())
  {
    mqttClient.connect(MQTT_CLIENT_ID);
  }
}


/* Function for disconnecting device 
   from MQTT broker and WLAN */
void CController::m_disconnect_from_MQTT()
{
  /* disconnect from MQTT broker */
  if(mqttClient.connected())
  {
    mqttClient.disconnect();
  }
}


/* Function for transmitting
   sensor data to MQTT broker */
void CController::m_transmitData()
{
  char payload[33];
  itoa(mi_pubData,payload,10);
  mqttClient.publish(MQTT_TOPIC, payload,true);
}



