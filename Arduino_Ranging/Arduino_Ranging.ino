#include <Arduino.h>
#include "Adafruit_ZeroTimer.h"
#include "Adafruit_ZeroFFT.h"

int powr = 2482;

// Frequency must be >= 800 Hz
float freq_timer = 4000.0; // 4000 Hz

// timer TC3
Adafruit_ZeroTimer zerotimer = Adafruit_ZeroTimer(3);
Adafruit_ZeroTimer zt4 = Adafruit_ZeroTimer(4);

void TC3_Handler() {
  Adafruit_ZeroTimer::timerHandler(3);
}

// the timer callback
// this is where the output power is set
bool trig_flag;
volatile bool count_up = true;

volatile int inc = 1241/((20e-3)*freq_timer); //voltage range/samples per up period
volatile int trig = 0;
void TimerCallback0(void)
{
  if(trig_flag)
  {
  // Ramp Function for inigration with a VCO
  if(count_up)
  {
    if(powr < 3723)
    {
      powr=powr+inc;
    }
    else
    {
      powr=powr-inc;
      count_up=false;
      trig = -1000;
    }
  }
  else
  {
    if ( powr > 2482)
    {
      powr=powr-inc;
    }
    else
    {
      powr=powr+inc;
      count_up = true;
      trig = 1000;
    }
  }
  //Wait until the dac it ready
  while (DAC->SYNCBUSY.bit.DATA0);
  // and write the data
  DAC->DATA[0].reg = powr;
  }
  else
  {
    trig = 0;
  }
}


int sar_pin = 0;

void setup()
{
    analogWrite(A0, powr); // initialize the DAC
    //digitalWrite(50, LOW);
    pinMode(sar_pin, INPUT);
    trig_flag = digitalRead(sar_pin);
    
    Serial.begin(115200);
    while(!Serial);                 // Wait for Serial monitor before continuing

    tc_clock_prescaler prescaler = TC_CLOCK_PRESCALER_DIV1;
    uint16_t compare = 48000000/freq_timer;
 
    zerotimer.enable(false);
    zerotimer.configure(prescaler,       // prescaler
              TC_COUNTER_SIZE_16BIT,       // bit width of timer/counter
              TC_WAVE_GENERATION_MATCH_PWM // frequency or PWM mode
              );
    
    zerotimer.setCompare(0, compare);
    zerotimer.setCallback(true, TC_CALLBACK_CC_CHANNEL0, TimerCallback0);
    zerotimer.enable(true);
}


int value = 0;

void loop()
{
  value = analogRead(A15);
  Serial.print(value);
  Serial.print(",");
  Serial.println(trig);

  trig_flag = digitalRead(sar_pin);
}
