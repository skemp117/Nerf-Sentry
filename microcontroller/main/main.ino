#define X_STP 2 // maps to azimuth
#define X_DIR 5

#define Y_STP 3 // maps to elevation
#define Y_DIR 6

#define Z_STP 4
#define Z_DIR 7

#define MAX_STEP_SPEED 4000 // Limit of accelstepper on arduino uno
#define ACCELRATION 10000

#define FULL_STEPS_PER_REV 200 // stepper motor specific
#define DEGREES_PER_REV 360
#define DEGREE_MULTIPLIER 3600 // conversion from degrees to arcsecs

#define AZ_DIRECTION -1 // pos goes left when this is negative
#define AZ_MICROSTEPS 8
#define AZ_GEAR_RATIO 144/17 // gear ratio converts from desired output to needed input, if we need 17 degrees, we need to move the stepper 144 degrees

#define EL_DIRECTION 1 // pos goes down when this is negative
#define EL_MICROSTEPS 8
#define EL_GEAR_RATIO 64/21

// Includes
#include <AccelStepper.h>

// Define some steppers and the pins the will use
AccelStepper az_stepper(AccelStepper::DRIVER, X_STP, X_DIR);
AccelStepper el_stepper(AccelStepper::DRIVER, Y_STP, Y_DIR); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5

long az = 0; // delta Azimuth in units of arcseconds (1/3600th of a degree)
long el = 0; // delta Elevation in units of arcseconds (1/3600th of a degree)
long ra = 0; // estimated range of target in mm
int fs = 0; // fire signal

bool send_serial = true;

void setup() {
  Serial.begin(115200); // Set the baud rate to match your communication rate
  // Clear serial buffer
  while(Serial.available() > 0) {
    char t = Serial.read();
  }

  az_stepper.setMaxSpeed(MAX_STEP_SPEED);
  az_stepper.setAcceleration(ACCELRATION);
  el_stepper.setMaxSpeed(MAX_STEP_SPEED);
  el_stepper.setAcceleration(ACCELRATION);
}

void loop(){
  getStepperCommands();
  az_stepper.run();
  el_stepper.run();
}

long convertAzToSteps(long az){
// az is the number of degrees we want to travel
  long az_steps = az * AZ_DIRECTION * AZ_GEAR_RATIO * AZ_MICROSTEPS * FULL_STEPS_PER_REV / DEGREES_PER_REV / DEGREE_MULTIPLIER; // gear ratio converts from desired output to needed input
  return az_steps;
}

long convertElToSteps(long el){
  long el_steps = el * EL_DIRECTION * EL_GEAR_RATIO * EL_MICROSTEPS * FULL_STEPS_PER_REV / DEGREES_PER_REV / DEGREE_MULTIPLIER; // gear ratio converts from desired output to needed input
  return el_steps;
}

void getStepperCommands(){
  long az_steps = 0;
  long el_steps = 0;
  if (Serial.available() > 0) {
    char buffer[32]; // Adjust the buffer size as needed
    int bytesRead = Serial.readBytesUntil('\n', buffer, sizeof(buffer) - 1);
    bool serial_update = false;
    buffer[bytesRead] = '\0'; // Null-terminate the string

    // Use strtok to split the message into 'a' and 'e' parts
    char *token = strchr(buffer, 'a');
    if (token != NULL) {
      token++;
      az = atol(token); // Convert the 'a' part to a long
      az_steps = convertAzToSteps(az);
      serial_update = true;
      az_stepper.move(az_steps);
    }

    token = strchr(buffer, 'e');
    if (token != NULL) {
      token++;
      el = atol(token); // Convert the 'e' part to a long
      el_steps = convertElToSteps(el);
      serial_update = true;
      el_stepper.move(el_steps);
    }

    token = strchr(buffer, 'r');
    if (token != NULL) {
      token++;
      ra = atol(token); // Convert the 'r' part to a long
    }

    token = strchr(buffer, 'f');
    if (token != NULL) {
      token++;
      fs = atoi(token); // Convert the 'r' part to a long
    }

    if (serial_update && send_serial) {
      Serial.print("Az steps: ");
      Serial.print(az_steps);
      Serial.print("\t El steps: ");
      Serial.print(el_steps);
      Serial.print("\t Range: ");
      Serial.print(ra);
      Serial.print("\t Fire: ");
      Serial.print(fs);
      Serial.println();
    }
  }
}
