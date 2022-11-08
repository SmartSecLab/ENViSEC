#include "Arduino.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <avr/io.h>



void disable_ports(void);

void setup()
{
  /* disable all input pins */
  disable_ports();

  /* enable emplyed I/O port (PD6:output) */ 
  PORTD_PIN6CTRL = PORT_PULLUPEN_bm 
                  | PORT_ISC_INPUT_DISABLE_gc;
  PORTD_DIRSET = PIN6_bm;                   //Set PD6 to output

  /* set up sleep mode */
  set_sleep_mode(SLEEP_MODE_PWR_DOWN);      //Set sleep mode to power down MCU

  /*disable BOD during sleep*/
  //BOD.CTRLA = BOD_SLEEP_DIS_gc;

  /* Enable interrupts globally and 
  configure PORTA.PIN2 for async interrupts */

  
  // FUSE.BODCFG = SLEEP_DIS_gc;
  //   CPU_CCP = CCP_IOREG_gc;
  // FUSE.BODCFG = 0;
  // BOD.CTRLA = 0;
  _PROTECTED_WRITE( BOD.CTRLA, 0 );  //disable BOD during sleep
  sei();                                          //enable global interrupts
  PORTA.PIN2CTRL = PORT_ISC_FALLING_gc            // sense falling edge on PORTA (ext-int)
                    | PORT_PULLUPEN_bm;           // Set PA2-state high
}


ISR(PORTA_PORT_vect)
{
  //ISR for wakeup
    PORTA.INTFLAGS = PIN2_bm;           //clear port interrupt flag
}

void loop()
{
  PORTD_OUTTGL = PIN6_bm;
  delay(5000);
  PORTD_OUTTGL = PIN6_bm;
  delay(5000);


  // /*go to sleep*/
  // sleep_enable();                     //enable sleep mode
  // sei();                              //set global interrupt flag for waking up
  // sleep_cpu();                        //sleep, wake up on PORTA, ISR execution, then execute line below
  // sleep_disable();                    //disable sleep mode
}

void disable_ports(void){
  for (volatile uint8_t* p = &PORTA.PIN0CTRL; p <= &PORTF.PIN0CTRL; p +=0x20)
  {
    for(uint8_t i = 0; i < 8; i++) p[i] = PORT_ISC_INPUT_DISABLE_gc; //pin dinput disable
  }
}



