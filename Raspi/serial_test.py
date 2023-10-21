# Importing Libraries 
import serial 
import time 

import serial.tools.list_ports
port = None
ports = list(serial.tools.list_ports.comports())
for p in ports:
    if p.manufacturer:
        if "Arduino" in p.manufacturer:
            port = p.device
            break

while not port:
    print("NO ARDUINO DETECTED")
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.manufacturer:
            if "Arduino" in p.manufacturer:
                port = p.device
                break
    time.sleep(0.1)

arduino = serial.Serial(port=port, baudrate=115200, timeout=1) 
arduino.write_timeout = 0.1


def write_read(x):
    arduino.write(bytes(x + '\n', 'utf-8')) 
    time.sleep(0.1) 
    val = arduino.readline()                # read complete line from serial output
    while not '\\n'in str(val):         # check if full data is received. 
        # This loop is entered only if serial read value doesn't contain \n
        # which indicates end of a sentence. 
        # str(val) - val is byte where string operation to check `\\n` 
        # can't be performed
        time.sleep(.001)                # delay of 1ms 
        temp = arduino.readline()           # check for serial output.
        if not not temp.decode():       # if temp is not empty.
            val = (val.decode()+temp.decode()).encode()
            # requrired to decode, sum, then encode because
            # long values might require multiple passes
    val = val.decode()                  # decoding from bytes
    val = val.strip()                   # stripping leading and trailing spaces.
    return val

arduino.reset_input_buffer()
arduino.reset_output_buffer()

while True: 
    num = input("Enter data: ") # Taking input from user 
    value = write_read(num) 
    print(value) # printing the value 
