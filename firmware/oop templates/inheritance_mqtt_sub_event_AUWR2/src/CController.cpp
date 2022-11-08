#include "CController.h"

/* Constructor */
CController::CController()//:Virtual_Sleep_Controller()
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
bool CController::m_virtualSleep(){}
bool CController::m_virtualTransmission(){}

/* Function for setting data member in abstract 
  parent class Virtual_Transmission_Controller */
void CController::m_setData(int data) {this->mi_data = data;}

/* Function for getting data member in abstract 
  parent class Virtual_Transmission_Controller */
int CController::m_getData() {return this->mi_data;}


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

/* Global declaration of object */
//CController controllerObj; 