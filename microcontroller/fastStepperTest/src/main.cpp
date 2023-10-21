#include "FastAccelStepper.h"

#define ENABLE_PIN 62
#define DIR_PIN   48
#define STEP_PIN  46

#define MICROSTEPS 16
#define STEPS_PER_REV 200 //1.8 degrees per step, 3200 steps per rev at 16 microsteps

#define MICROSTEPS_PER_REV MICROSTEPS*STEPS_PER_REV

int step_cnt = 0;

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;

unsigned long lastMillis = 0;
int newRPM = 1;

void setup() {
    Serial.begin(112500);
    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (stepper) {
        stepper->setEnablePin(ENABLE_PIN);
        stepper->setAutoEnable(true);
        stepper->setDirectionPin(DIR_PIN);

        Serial.print("Setting speed to: ");
        Serial.print(newRPM);
        Serial.println(" RPM");
        
        stepper->setSpeedInHz(newRPM*3200);       // steps/s
        stepper->setAcceleration(1000);    // steps/s²
        stepper->runForward();
    }
}

void loop() {
    // stepper->move(3200*10);
    if (millis()-lastMillis > 10000){
        newRPM++;
        stepper->setSpeedInHz(newRPM*3200); 
        stepper->runForward();
        Serial.print("Setting speed to: ");
        Serial.print(newRPM);
        Serial.println(" RPM");
        lastMillis = millis();
    }
}