#define X_STP 2
#define Y_STP 3
#define Z_STP 4

#define X_DIR 5
#define Y_DIR 6
#define Z_DIR 7

#define FULL_STEPS_PER_REV 200
#define MICROSTEPS_PER_STEP 2
#define RPS_TO_STEPS  FULL_STEPS_PER_REV*MICROSTEPS_PER_STEP
#define MAX_STEPS_PER_SEC 5*RPS_TO_STEPS
#define MAX_STEP_SPEED 4000


#include <AccelStepper.h>

// Define some steppers and the pins the will use
AccelStepper stepper1(AccelStepper::DRIVER, Y_STP, Y_DIR); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5



void setup()
{  
    Serial.begin(115200);

    stepper1.setMaxSpeed(MAX_STEP_SPEED);
    stepper1.setAcceleration(1000.0);
    stepper1.moveTo(10*MAX_STEPS_PER_SEC);
}

void loop()
{
    // Change direction at the limits
    if (stepper1.distanceToGo() == 0){
	    stepper1.moveTo(-stepper1.currentPosition());
      Serial.println("switch");
    }
    stepper1.run();
    // stepper1.runSpeed();
}