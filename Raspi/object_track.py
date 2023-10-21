import collections
import argparse

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from picamera2 import Picamera2
import cv2
import numpy as np

from sort import *

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
arduino.write_timeout = 0
arduino.reset_input_buffer()
arduino.reset_output_buffer()

def write_read(x):
    arduino.write(bytes(x + '\n', 'utf-8')) 
    arduino.reset_input_buffer()
    return x    
    # time.sleep(0.1) 
    # val = arduino.readline()                # read complete line from serial output
    # while not '\\n'in str(val):         # check if full data is received. 
    #     # This loop is entered only if serial read value doesn't contain \n
    #     # which indicates end of a sentence. 
    #     # str(val) - val is byte where string operation to check `\\n` 
    #     # can't be performed
    #     time.sleep(.001)                # delay of 1ms 
    #     temp = arduino.readline()           # check for serial output.
    #     if not not temp.decode():       # if temp is not empty.
    #         val = (val.decode()+temp.decode()).encode()
    #         # requrired to decode, sum, then encode because
    #         # long values might require multiple passes
    # val = val.decode()                  # decoding from bytes
    # val = val.strip()                   # stripping leading and trailing spaces.
    # return val



def draw_objects(img, detections):
  """Draws the bounding box and label for each object."""
  for element in detections:
    cv2.rectangle(img, 
                  (int(element[1]), int(element[0])), 
                  (int(element[3]), int(element[2])), 
                  (255,0,0), 1)
    cv2.putText(img, f'id: {int(element[4]):d}',
                (int(element[1]) + 10, int(element[0]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,0,0),1,cv2.LINE_AA)
    
def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 30.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)
# src_size = (1296,972)
src_size = (640,480)
# src_size = (300,300)
picam_fov = np.array((54,41)) # degrees
mid_pt = np.array(src_size)/2

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": src_size, "format": "RGB888"})
picam2.configure(video_config)
picam2.start()

threshold = 0.4
model = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
labels = read_label_file('coco_labels.txt')
inference_size = (300,300)
person_label_id = 0

interpreter = make_interpreter(model)
interpreter.allocate_tensors()

#create instance of SORT
mot_tracker = Sort() 

fps_counter = avg_fps_counter(30)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 25)
fontScale = 1
color = (0, 255, 0)
thickness = 2

def get_objects(interpreter,
                score_threshold,
                scale):

    boxes = common.output_tensor(interpreter, 0)[0]
    class_ids = np.uint8(common.output_tensor(interpreter, 1)[0])
    scores = common.output_tensor(interpreter, 2)[0]
    count = int(common.output_tensor(interpreter, 3)[0])

    width, height = common.input_size(interpreter)
    image_scale_x, image_scale_y = scale
    sx, sy = width / image_scale_x, height / image_scale_y

    def make(i):
        if class_ids[i] != person_label_id:
            return None
        return list((boxes[i]*np.array([sy, sx, sy, sx])).astype(int))+[scores[i]] #ymin, xmin, ymax, xmax

    return np.array([element for i in range(count) if (element:=make(i)) is not None and scores[i] >= score_threshold])

while True:
    full_image = cv2.flip(picam2.capture_array("main"),0)
    image = cv2.resize(full_image, inference_size, interpolation=cv2.INTER_NEAREST)
    scale = tuple(a/b for a,b in zip(inference_size, src_size))
    common.set_input(interpreter, image)

    start = time.perf_counter()
    interpreter.invoke()
    detections = get_objects(interpreter, threshold, scale)
    inference_time = time.perf_counter() - start

    if detections.size == 0:
        print('No objects detected')  
    else:
        track_bbs_ids = mot_tracker.update(detections) # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
        if not track_bbs_ids.any():
            track_bbs_ids = detections
        draw_objects(full_image, track_bbs_ids)
        center_pts = np.transpose(
        np.array([np.mean(track_bbs_ids[:,[1, 3]], axis = 1), 
                np.mean(track_bbs_ids[:,[0, 2]], axis = 1)])) # (+right, +down)
        
        dist_to_center = np.linalg.norm(center_pts-mid_pt,axis=1) 
        min_dist_idx = np.argmin(dist_to_center)

        delta_az_el_pixels = (center_pts[min_dist_idx]-mid_pt) # (+right, +down)
        delta_az_el_degrees = picam_fov/src_size*delta_az_el_pixels
        delta_az_el_arcsecs = delta_az_el_degrees*3600

        delta_az_el_str = f"a{delta_az_el_arcsecs[0]:.0f}e{delta_az_el_arcsecs[1]:.0f}"
        print(write_read(delta_az_el_str))

    av_fps = next(fps_counter)
    text = f'FPS: {av_fps:.1f} T_i: {inference_time*1000:.1f}ms'
    image = cv2.putText(full_image, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Camera", full_image)
    if cv2.waitKey(1)==ord('q'):
        break