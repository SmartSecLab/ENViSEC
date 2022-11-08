#include "Arduino.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <avr/io.h>
#include <xmega.h>


#define OPERATION_TIME_PERIOD_MINUTES 1            //Numbers of run
#define INTERRUPT_PIN 

//  #include "avr8-stub.h"
//  #include "app_api.h" // only needed with flash breakpoints

void debugISR();

void setup() {
  //Serial for debugging
  //  Serial.begin(9600);

  //Disable all I/O-ports
  for (volatile uint8_t* p = &PORTA.PIN0CTRL; p <= &PORTF_PIN0CTRL; p +=0x20)
  {
    for(uint8_t i = 0; i < 8; i++) p[i] = PORT_ISC_INPUT_DISABLE_gc;
  }

  //Setup employed I/O port 
  PORTD_PIN6CTRL |= PORT_PULLUPEN_bm;
  PORTD_DIRSET |= PIN6_bm;  //Set PD6 to output
  PORTA.DIRSET &= ~(1<<0);  //Clear bit to set as output
  PORTA.PIN0CTRL |= PORT_PULLUPEN_bm;
  
 sleep_bod_disable();
 



/*--------------------------------------------
  //Set up Periodic interrupt
  RTC_PITCTRLA |= RTC_PERIOD_CYC32768_gc |  //Set period length for periodic interrupt (1 sec)
                  RTC_RTCEN_bm;             //Enable RTC peridoc interrupt
  RTC_PITINTCTRL |= RTC_PI_bm;              //PI, pit interrupt enable

  //Set up BOD disable during sleep
  CPU_CCP = CCP_IOREG_gc;                   //Sign. to enable setting BOD sleep mode bits
  BOD_CTRLA &= ~BOD_SLEEP_gm;               //Clearing BOD sleep mode bitfields [1:0] to disable BOD during sleep
  BOD_CTRLA |= BOD_ACTIVE_ENWAKE_gc;        //Halt wake-up until BOD is ready


  //debug external interrupt
  pinMode(2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(2), debugISR, LOW);
  ----------------------------------------------*/
}

// void debugISR(){
// PORTD_OUTTGL |= PIN6_bm;

// }

ISR(RTC_PIT_vect)
{

  RTC_PITINTFLAGS |= RTC_PITEN_bm;

  //check if time for task execution 
    PORTD_OUTTGL |= PIN6_bm;
  }
}

// ISR(RTC_PIT_vect)
// {
//   static int cyc_Cntr = 0;

//   RTC_PITINTFLAGS |= RTC_PITEN_bm;
//   cyc_Cntr++;

//   //check if time for task execution 
//   if(cyc_Cntr >= OPERATION_TIME_PERIOD_MINUTES * 60){
//     //Code for task execution here
//     PORTD_OUTTGL |= PIN6_bm;

//     //reset cycle counter after task completed
//     cyc_Cntr = 0;
//   }
// }



 
void loop() {


}



  //WDT setup
  // CCP |= CCP_IOREG_gc;                    //Eable write access to setup WDT
  // WDT_CTRLA |= WDT_PERIOD_8KCLK_gc;       //Setup WDT for 8 seconds timout
