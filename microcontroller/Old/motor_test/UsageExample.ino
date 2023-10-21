#include "FastAccelStepper.h"

// As in StepperDemo for Motor 1 on AVR
//#define dirPinStepper    5
//#define enablePinStepper 6
//#define stepPinStepper   9  // OC1A in case of AVR

// As in StepperDemo for Motor 1 on ESP32

#define dirPinStepper 55
#define enablePinStepper 38
#define stepPinStepper 54

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;

void setup() {
  engine.init();
  stepper = engine.stepperConnectToPin(stepPinStepper);
  Serial.begin(115200);
  Serial.println("Ooof");
  if (stepper) {
    Serial.println("HI");
    stepper->setDirectionPin(dirPinStepper);
    stepper->setEnablePin(enablePinStepper);
    stepper->setAutoEnable(true);

    // If auto enable/disable need delays, just add (one or both):
    // stepper->setDelayToEnable(50);
    // stepper->setDelayToDisable(1000);

    stepper->setSpeedInUs(1000);  // the parameter is us/step !!!
    stepper->setAcceleration(100);
    stepper->move(1000);
  }
}

void loop() {}
