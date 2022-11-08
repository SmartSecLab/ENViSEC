/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "output_handler.h"

#include "Arduino.h"
#include "constants.h"

// The pin of the Arduino's built-in LED
int led = LED_BUILTIN;

// Track whether the function has run at least once
bool initialized = false;

// Track previous inference outputs...
int prev_len = 3;
bool prev_out[4];
bool prev_status;

// Animates a dot across the screen to represent the current x and y values
void HandleOutput(tflite::ErrorReporter* error_reporter, float y1, float y2) {
  // Do this only once
  if (!initialized) {
    // Set the LED pin to output
    pinMode(led, OUTPUT);
    initialized = true;
  }
  bool output = (y1 == 0) ? LOW : HIGH; // we'll use x_2 as we wanna focus on 'LED ON'.
  // if the previous outputs are the same as the current, then the LED is likely on...
  // this is to remove noisy false outputs from the model...
  // if (prev_out[0] == prev_out[1] && prev_out[1] == output){

  // if (output==LOW || prev_out[1]==HIGH)
  // // if (output!=prev_out[1])
  // {
  //   digitalWrite(led, LOW);
  //     TF_LITE_REPORT_ERROR(error_reporter, "LED guess: %d", LOW);
  // }
  // if (output==HIGH || prev_out[1]==LOW)
  // {
  //   digitalWrite(led, HIGH);
  //     TF_LITE_REPORT_ERROR(error_reporter, "LED guess: %d", HIGH);
  // }

  // if (prev_out[0]!=output || prev_out[0]!=prev_out[1] || prev_out[1] != output)
  // if (prev_out[0] == prev_out[1]  && prev_out[0]==prev_out[2] && prev_out[1]==prev_out[2] && prev_out[2] == output)

  for(int i=0; i<prev_len; i++)
  {
    prev_status = (prev_out[i]==prev_out[i+1])? LOW:HIGH;
  }
  if (prev_status==LOW && prev_out[prev_len]==output)
  {
    digitalWrite(led, LOW);
    TF_LITE_REPORT_ERROR(error_reporter, "LED guess: %d", LOW);
  }
  else
  {
    digitalWrite(led, HIGH);
    TF_LITE_REPORT_ERROR(error_reporter, "LED guess: %d", HIGH);
    prev_status=HIGH;
  }
  // swapping the output the previous states
  for(int i=0; i<prev_len; i++)
  {
    prev_out[i]=prev_out[i+1];
  }
  prev_out[prev_len]=output;
}