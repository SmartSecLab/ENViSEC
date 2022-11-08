#include "Controller.h"


/* Constructor */
Controller::Controller()
{
  //this->m_disable_ports();
  /* debug: leds */
  PORTD_PIN6CTRL |= PORT_PULLUPEN_bm;
  PORTD_DIRSET |= PIN6_bm;  //Set PD6 to output
  PORTD_PIN0CTRL |= PORT_PULLUPEN_bm;
  PORTD_DIRSET |= PIN0_bm;  //Set PD0 to output
};


/**************************************************
*                MEMBER FUNCTIONS                 *
**************************************************/

/* Virtual overriding functions
   with no functionality */
bool Controller::m_virtualSleep(){}
bool Controller::m_virtualTransmission(){}

/* Function for setting data member in abstract 
  parent class Virtual_Transmission_Controller */
void Controller::m_setData(int data) {this->mi_data = data;}

/* Function for getting data member in abstract 
  parent class Virtual_Transmission_Controller */
int Controller::m_getData() {return this->mi_data;}



/* Function for controlling the process of the device.
  This function is called in main and uses members from 
  parent classes to determine when to sleep, wake up, 
  and transmit data.*/
void Controller::m_processController()
{
  if(this->m_checkForWakeup())          //scope: AbstractSleepController 
  {
    mainFunctionallity();               //scope: Global
    if(this->m_ThresholdCheck())        //scope: Local
    {
      this->m_connect_to_MQTT();        //scope: AbstractTransmissionController
      this->m_transmitData();           //scope: AbstractTransmissionController 
      this->m_disconnect_from_MQTT();   //scope: AbstractTransmissionController
      this->m_deviceSleep();            //scope: AbstractSleepController
    } 
  }
  this->m_deviceSleep();
}


bool Controller::m_ThresholdCheck()
{
  if(this->mi_data > 400 /*this->mi_upperThreshold*/ || this->mi_data < 399/*this->mi_lowerThreshold*/) return true;
  else return false;
}

/* Function for disabling ports on devicet 
  parent class Virtual_Transmission_Controller */
void Controller::m_disablePorts(void){
  for (volatile uint8_t* p = &PORTA.PIN0CTRL; p <= &PORTF.PIN0CTRL; p +=0x20)
  {
    for(uint8_t i = 0; i < 8; i++) p[i] = PORT_ISC_INPUT_DISABLE_gc; //pin input disable
  }
}

/* Global declaration of object */
Controller controllerObj; 