import time 
import collections

from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from picamera2 import Picamera2
import cv2
import numpy as np

from sort import *

import serial 
import serial.tools.list_ports

import multiprocessing
import copy

# Configuration variables
DISPLAY = True
# Set hue values of friend color (0-15 and 165-179)
FRIEND_HUE = lambda x : np.logical_or((x <= 10), (x >= 160))
# FRIEND_HUE = np.concatenate((np.arange(0, 16, 1, dtype=int), np.arange(165, 181, 1, dtype=int)), axis = 0)
# Set saturation values of friend color (50-255)
# FRIEND_SAT = np.arange(50, 256, 1, dtype=int)
FRIEND_SAT = lambda x : (x >= 100)
FRIEND_VAL = lambda x : (x >= 20)
FRIEND_THRESHOLD = 0.20 # expressed as a ratio (i.e. .2 is 20%)

SENSOR_VERTICAL_SIZE = 0.0024 # sensor height in m
REFERENCE_HEIGHT = 1.5 # m height of target object
FOCAL_LENGTH = 0.0036 # m focal length

TARGET_TIMEOUT = 15 # target timeout in seconds
FIRE_DURATION = 5 # length of time it will fire at a target

SRC_SIZE = (640,480)

# Functions
def draw_objects(img, detections, color):
  """Draws the bounding box and label for each object."""
  for element in detections:
    cv2.rectangle(img, 
                  (int(element[1]), int(element[0])), 
                  (int(element[3]), int(element[2])), 
                  color, 1)
    cv2.putText(img, f'id: {int(element[4]):d}',
                (int(element[1]) + 10, int(element[0]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                color,1,cv2.LINE_AA) 
      
def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.perf_counter()
    yield 30.0  # First fps value.

    while True:
        curr = time.perf_counter()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

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

class TargetList:
    def __init__(self, max_elements):
        self.target_ids = collections.deque(maxlen=max_elements)
        self.target_last_times = collections.deque(maxlen=max_elements)
    
    def add_target(self, target_id):
        if target_id not in self.target_ids:
            self.target_ids.append(target_id)
            self.target_last_times.append(time.perf_counter())
            return True # this is a new target, it is a valid foe
        else:
            index = self.target_ids.index(target_id)
            time_since = time.perf_counter() - self.target_last_times[index]
            if time_since > (TARGET_TIMEOUT + FIRE_DURATION): # if the elapsed time is greater than the timeout
                self.target_last_times[index] = time.perf_counter()
                return True # timeout has passed, this target id is a valid foe
            elif time_since < FIRE_DURATION: # if less time has passed than the duration of fire, keep firing
                return True
            else:
                return False # timeout is active, this target id is not valid

target_list = TargetList(10)

def select_foe(foes):
    if foes.shape[0] == 0:
        return foes, None

    # # filter out foes that are on timeout
    # for foe_temp in foes:
    #     target_list.add_target(foe_temp)

    center_pts = np.transpose(
    np.array([np.mean(foes[:,[1, 3]], axis = 1), 
            np.mean(foes[:,[0, 2]], axis = 1)])) # (+right, +down)
    
    dist_to_center = np.linalg.norm(center_pts-mid_pt,axis=1) 
    min_dist_idx = np.argmin(dist_to_center)
    center_pt = center_pts[min_dist_idx]
    
    delta_az_el_arcsecs = get_delta_angles(center_pt)

    foe = np.empty((0,foes.shape[1]))
    foe = np.vstack((foe,foes[min_dist_idx])) # this is just setup to preserve indexing rules for the draw objects
    target_params = np.hstack((delta_az_el_arcsecs, get_range(foe[0][2] - foe[0][0])))
    
    return foe, target_params

def get_delta_angles(center_pt):

    delta_az_el_pixels = (center_pt-mid_pt) # (+right, +down)
    delta_az_el_degrees = picam_fov/SRC_SIZE*delta_az_el_pixels
    delta_az_el_arcsecs = delta_az_el_degrees*3600
    return delta_az_el_arcsecs

def get_range(pixel_height):
    return REFERENCE_HEIGHT * FOCAL_LENGTH / (pixel_height * (SENSOR_VERTICAL_SIZE / SRC_SIZE[1]))
def friend_or_foe(image, bbs):
    foes = np.zeros((0,bbs.shape[1]))
    friends = np.zeros((0,bbs.shape[1]))
    # convert bounded image to HSV
    for bb in bbs:
        # create mask that fits in  H
        x1 = int(max(bb[1],0))
        x2 = int(min(bb[3],SRC_SIZE[0]))
        y1 = int(max(bb[0],0))
        y2 = int(min(bb[2],SRC_SIZE[1]))
        cropped_img = image[y1:y2, x1:x2]
        # cropped_img = image
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV) # why is it bgr when I initally made it RGB? Who knows
        # calculate percent area of mask to total
        mask = np.logical_and(FRIEND_HUE(hsv_img[:,:,0]), FRIEND_SAT(hsv_img[:,:,1]), FRIEND_VAL(hsv_img[:,:,2]))
        # image[mask] = (0,0,255)
        # cv2.imshow("Camera", image)
        fraction_friend = np.sum(mask)/np.size(hsv_img[:,:,0])
        # if above threshold return true
        # print(fraction_friend)
        if fraction_friend < FRIEND_THRESHOLD: # if we are not friend, add to foe list
            foes = np.vstack((foes,bb))
        else:
            friends = np.vstack((friends,bb))
    return friends, foes

def fire_timeout():
    # check list of targets for target ID
    # if target is on list
        # check if now minus last fire time exceeds fire timeout
            # if yes return fire
            # if no return don't fire
    # else
        # add target ID to list
        # set fire signal to true
        # add last fired time to list
    pass

def write_to_arduino(target_params):
    # a: azimuth in arcseconds
    # e: elevation in arcseconds
    # r: range in meters
    # f: fire boolean
    x = f"a{target_params[0]:.0f}e{target_params[1]:.0f}"
    arduino.write(bytes(x + '\n', 'utf-8')) 
    arduino.reset_input_buffer()
    return x    



# Setup code
if True:
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

    arduino = serial.Serial(port=port, baudrate=115200, timeout=1, rtscts=True, write_timeout=True) 
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    
    picam_fov = np.array((54,41)) # degrees
    mid_pt = np.array(SRC_SIZE)/2

    threshold = 0.4
    
    labels = read_label_file('coco_labels.txt')
    inference_size = (300,300)
    person_label_id = 0

    model = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
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

def get_image_and_infer():
    while True:
        # Get image
        full_image = p1_queue.get()
        
        # input to model
        image = cv2.resize(full_image, inference_size, interpolation=cv2.INTER_NEAREST)
        scale = tuple(a/b for a,b in zip(inference_size, SRC_SIZE))
        common.set_input(interpreter, image)

        # Perform Inference
        interpreter.invoke()
        detections = get_objects(interpreter, threshold, scale)
        data = (full_image, detections)
        process_inference_queue.put(data)


def process_inference():
    while True:
        data = process_inference_queue.get()
        full_image, detections = data
        # Extract results
        if detections.size == 0:
            print('No objects detected') 
        else:
            track_bbs_ids = mot_tracker.update(detections) # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
            if not track_bbs_ids.any():
                track_bbs_ids = detections
                track_bbs_ids[:,4] = 0 # this makes all the ids zero so there isn't some weird float thing going on when the tracker doesn't update

            friends, foes = friend_or_foe(full_image, track_bbs_ids)
            foe, target_params = select_foe(foes)
            print(target_params)

        # print(get_range(foes[0][2] - foes[0][0]))
        draw_objects(full_image, friends, (0,255,0))
        draw_objects(full_image, foes, (255,0,0))
        draw_objects(full_image, foe, (0,0,255))

        write_to_arduino(target_params)

        av_fps = next(fps_counter)

        text = f'FPS: {av_fps:.1f}'
        cv2.putText(full_image, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        if DISPLAY:
            cv2.imshow("Camera", full_image)
        if cv2.waitKey(1)==ord('q'):
            break


if __name__ == "__main__":
    p1_queue = multiprocessing.Queue(maxsize=1)
    process_inference_queue = multiprocessing.Queue(maxsize=1)

    thread1 = multiprocessing.Process(target = get_image_and_infer)
    thread2 = multiprocessing.Process(target = process_inference)

    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"size": SRC_SIZE, "format": "RGB888"},
        controls={'NoiseReductionMode': 0},
        buffer_count=6 , queue=True)
    picam2.configure(video_config)
    picam2.start()

    # Start the threads
    thread1.start()
    thread2.start()

    while True:
        full_image = cv2.flip(picam2.capture_array("main"),0)
        p1_queue.put(copy.deepcopy(full_image))



    
    
    


    
    