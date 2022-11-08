#include "CASleepController.h"

/* Constructor */
CASleepController::CASleepController()
{
  /* Sleep mode setup */
  set_sleep_mode(SLPCTRL_SMODE_PDOWN_gc);

  /* Wake up timer setup (PIT) */
  RTC_PITCTRLA = RTC_PERIOD_CYC32768_gc    //Set period length for periodic interrupt (1 sec)
                  | RTC_RTCEN_bm;           //Enable RTC peridoc interrupt
  RTC_PITINTCTRL = RTC_PI_bm;              //PIT interrupt enable
}


/**************************************************
*                MEMBER FUNCTIONS                 *
**************************************************/

/* Function for checking for device 
   wake up, wakeing up involves commencing 
   device's Main Functionality and transmission
   of data to broker */
bool CASleepController::m_checkForWakeup()
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
void CASleepController::m_deviceSleep()
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
void CASleepController::m_incrementSleepCounter()
{
  this->mi_sleepCntr++;
}


/* Function for resetting sleep counter after 
   transmission of data to MQTT broker*/
void CASleepController::m_resetSleepCounter()
{
  this->mi_sleepCntr = 0;
}


/* Function for allowing external interrupts
  to make the PIT wake up the device from sleep*/
ISR(RTC_PIT_vect)
{
  RTC_PITINTFLAGS |= RTC_PITEN_bm;
}