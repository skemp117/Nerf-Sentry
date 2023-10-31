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

#define BUFFER_SIZE 100
#define BUFFER_RESOLUTION 1000 // buffer updates every ___ microseconds

#define SEND_SERIAL false

#define BAUD_RATE 115200
#define SERIAL_BUF_SIZE 64
#define TRANS_DELAY SERIAL_BUF_SIZE*1000000/BAUD_RATE

#define P_GAIN 0.5

// Includes
#include <AccelStepper.h>

// Define some steppers and the pins the will use
AccelStepper az_stepper(AccelStepper::DRIVER, X_STP, X_DIR);
AccelStepper el_stepper(AccelStepper::DRIVER, Y_STP, Y_DIR); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5

long az = 0; // delta Azimuth in units of arcseconds (1/3600th of a degree)
long el = 0; // delta Elevation in units of arcseconds (1/3600th of a degree)
long ra = 0; // estimated range of target in mm
unsigned long dt = 0; // time offset in microseconds
int fs = 0; // fire signal

unsigned long timeMicrosBuffer[BUFFER_SIZE]; // buffering positions while motors are running
int head = 0;
int tail = 0;

long azPosBuffer[BUFFER_SIZE]; // buffering positions while motors are running
long elPosBuffer[BUFFER_SIZE]; // buffering positions while motors are running

unsigned long time_last = 0;

void setup() {
  Serial.begin(BAUD_RATE); // Set the baud rate to match your communication rate
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
  if (micros()-time_last > BUFFER_RESOLUTION){
    addToTimeBuffer(timeMicrosBuffer, head, micros()); // note micros wraps at 70 minutes
    addToBuffer(azPosBuffer, head, az_stepper.currentPosition());
    addToBuffer(elPosBuffer, head, el_stepper.currentPosition());
    time_last = micros();
  }
}

int getPosIdAtTime(long timeDelta){
  int id = head;
  unsigned long timeNow = 0;
  while (id != tail) {
    id = (id - 1) % BUFFER_SIZE;
    timeNow = micros()-timeMicrosBuffer[id];
    if (timeNow >= timeDelta){
      break;
    }
  }
  return id;
}

// Function to add data to the FIFO buffer
void addToTimeBuffer(unsigned long* buffer, int& head, long data) {
  buffer[head] = data;
  head = (head + 1) % BUFFER_SIZE; // Wrap around if the buffer is full
}

void addToBuffer(long* buffer, int& head, long data) {
  buffer[head] = data;
  head = (head + 1) % BUFFER_SIZE; // Wrap around if the buffer is full
}

// Function to remove and return data from a specified buffer
int removeFromBuffer(int* buffer, int& tail) {
  int data = buffer[tail];
  tail = (tail + 1) % BUFFER_SIZE;
  return data;
}

// Function to check if a specified buffer is empty
bool isBufferEmpty(int head, int tail) {
  return head == tail;
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
    char buffer[SERIAL_BUF_SIZE]; // Adjust the buffer size as needed
    int bytesRead = Serial.readBytesUntil('\n', buffer, sizeof(buffer) - 1);
    bool serial_update = false;
    buffer[bytesRead] = '\0'; // Null-terminate the string

    // Use strtok to split the message into 'a' and 'e' parts
    char *token = strchr(buffer, 'a');
    if (token != NULL) {
      token++;
      az = atol(token); // Convert the 'a' part to a long
      serial_update = true;
    }

    token = strchr(buffer, 'e');
    if (token != NULL) {
      token++;
      el = atol(token); // Convert the 'e' part to a long
      serial_update = true;
    }

    token = strchr(buffer, 'r');
    if (token != NULL) {
      token++;
      ra = atol(token); // Convert the 'r' part to a long
      serial_update = true;
    }

    token = strchr(buffer, 'f');
    if (token != NULL) {
      token++;
      fs = atoi(token); // Convert the 'r' part to a long
      serial_update = true;
    }

    token = strchr(buffer, 'o');
    if (token != NULL) {
      token++;
      dt = atol(token); // Convert the 'o' part to a long
      serial_update = true;
    }

    if (serial_update) {
      az = P_GAIN*az;
      el = .1*el;

      
      int id = getPosIdAtTime(dt+TRANS_DELAY);
      long az_offset = az_stepper.currentPosition() - azPosBuffer[id];
      long el_offset = el_stepper.currentPosition() - elPosBuffer[id];
      
      az_steps = convertAzToSteps(az) - az_offset;
      el_steps = convertElToSteps(el) - el_offset;

      az_stepper.move(az_steps);
      el_stepper.move(el_steps);
  
      if (SEND_SERIAL) {
        Serial.print("Az steps: ");
        Serial.print(az_steps);
        Serial.print("\tEl steps: ");
        Serial.print(el_steps);
        Serial.print("\tRange: ");
        Serial.print(ra);
        Serial.print("\tFire: ");
        Serial.print(fs);
        Serial.print("\tOffset: ");
        Serial.print(dt);
        Serial.print("\tO_id: ");
        Serial.print(id);
        Serial.println();
      }
    }
  }
}
