import cv2
import numpy as np
import time
from picamera2 import Picamera2
import threading
import serial
import v4l2
import csv
import psutil

######################### Record data ########################## 
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

def updateLog(t,h,dx,dy,X,Y,FPS):
    list_append = [{'t':'{:.04f}'.format(t),'h':'{:.02f}'.format(h),
                    'dx': '{:.02f}'.format(dx), 'dy': '{:.02f}'.format(dy), 
                    'X': '{:.02f}'.format(X), 'Y': '{:.02f}'.format(Y), 
                    'FPS': '{}'.format(FPS)}]
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerows(list_append)
        csvfile.close()
################################################################

######################### Receive height value from STM32 #########################
# # Init USB communication
# port = '/dev/ttyUSB0'  
# baudrate = 115200  
# try:
#     ser = serial.Serial(port, baudrate, timeout=1)
# except serial.SerialException as e:
#     print(f"Error opening serial port: {e}")
#     exit(1)
# samples = []
# sample_num = 5

# Receive data from COM port
def receive_data():
    global camera_height
    # while True:
    #     if ser.in_waiting > 0:
    #         read_data = ser.readline().decode('utf-8').rstrip()
    #         try:
    #             config = -(float(read_data)*14/100) #Relative error = 14%
    #             samples.append(float(read_data)+ config)
    #             if len(samples) == sample_num:
    #                 camera_height = round( (sum(samples) / len(samples) / 10 ), 1)
    #                 samples.pop(0)
    #             else:
    #                 camera_height = (round( float(read_data) + config, 1) / 10)
    #         except:
    #             if len(samples) == sample_num:
    #                 camera_height = samples[-1]
    #             else:
    #                 continue
    #     time.sleep(0.01)
    camera_height = 78
##################################################################################

######################### Camera setting function #########################
def preprocess_frame(frame):
    width_crop = 100
    height_crop = 100
    height, width, _ = frame.shape
    # Split left, right image
    frameL = frame[:, :width // 2]
    frameR = frame[:, width // 2:]  
    heightL, widthL,_ = frameL.shape
    # Take center image
    center_x = widthL // 2
    center_y = heightL // 2
    # Take position for crop
    start_x = center_x - (width_crop // 2)
    start_y = center_y - (height_crop // 2)
    # Take crop image
    frameL = frameL[start_y:start_y + height_crop, start_x:start_x + width_crop]
    frameR = frameR[start_y:start_y + height_crop, start_x:start_x + width_crop]
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    return grayL
   
# Frame capture thread
def capture_frames():
    global frame_buffer
    global fps_cam, fps_cam_buffer
    t0cam = 0
    fps_limit = 60
    while True:
        # Guard CPU performance
        if fps_limit > 20 and psutil.cpu_percent() > 90:
            fps_limit -= 10
        elif fps_limit < 60 and psutil.cpu_percent() < 50:
            fps_limit += 10  
        if psutil.cpu_percent() > 90:
            time.sleep(0.01)
        # Take frame
        frame = picam2.capture_array()
        with buffer_lock:
            frame_buffer = frame
            t1cam = time.time()
            fps_cam_buffer = 1/(t1cam -t0cam)
            if (fps_cam > fps_limit):
                time.sleep(round(1/fps_limit - 1/fps_cam, 3))
            t0cam = t1cam


##################################################################################

######################### Initial function and parameter #########################

# Initialize the camera
picam2 = Picamera2(0)

picam2.preview_configuration.main.size = (1600, 400)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 150.0
picam2.preview_configuration.align()

picam2.configure("preview")

picam2.start()


# Lucas-Kanade parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Detect feature points parameters
feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters to find the cm per pixel 
width_img = 400
height_img = 1600
div_crop = 1
crop_size_w = 300
crop_size_h = 300
focal_length = 0.14
pixel_dimension = 3 * 10**-4 
sensor_pxl_w = 720
sensor_pxl_h = 2560
# cm_per_pxl_w_coef = (sensor_pxl_w * pixel_dimension / focal_length) / width_img 
# cm_per_pxl_h_coef = (sensor_pxl_h * pixel_dimension / focal_length) / height_img 
cm_per_pxl_w_coef = 1.9125*10**-3
cm_per_pxl_h_coef = 1.9125*10**-3
# Timing variables
fps = 0
fps_cam = 0
count_time = 0
count_print = 0
cB_count = 0
dx_cm = 0
dy_cm = 0
global_x = 0
global_y = 0
old_gray = None
p0 = None
prev_frame_time = 0
new_frame_time = 0
kalman = 0.9

# Shared buffer for frames
frame_buffer = None
fps_cam_buffer = None
buffer_lock = threading.Lock()

# Start frame capture thread
update_task = True
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Start data receiving thread
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()

prev_frame_time = time.time()
##############################################################################

######################### Main processing loop ###############################
prev_frame_time = time.time()
while True:
    with buffer_lock:
        if frame_buffer is not None:
            frame = frame_buffer
            fps_cam = 0.99 * fps_cam + 0.01*fps_cam_buffer
            frame = preprocess_frame(frame)
            frame_buffer = None
            fps_cam_buffer = None
        else:
            continue
    
    if old_gray is None:
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame, p0, None, **lk_params)
    except:
        for i in range(10):
            print("False tracking!!!!!!!!!!!!")
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        global_x += dx_cm
        global_y += dy_cm
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # Show detail of flow if need
    # sFrame = frame.copy()
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     cv2.line(sFrame, (a, b), (c, d), 255, 1)
    #     cv2.circle(sFrame, (a, b), 3, 255, -1)
    
    if len(good_new) > 0:
        dx_pixels = np.mean(good_old[:, 0] - good_new[:, 0])
        dy_pixels = np.mean(good_old[:, 1] - good_new[:, 1])

        cm_per_pxl_w = cm_per_pxl_w_coef * camera_height
        cm_per_pxl_h = cm_per_pxl_h_coef * camera_height
        
        if abs(dx_pixels * cm_per_pxl_w) >= 0*cm_per_pxl_w/7: 
            dx_cm = (1 - kalman) * dx_cm + kalman * dx_pixels * cm_per_pxl_w
        else: dx_cm = 0
        if abs(dy_pixels * cm_per_pxl_h) >= 0*cm_per_pxl_h/7: 
            dy_cm = (1 - kalman) * dy_cm + kalman * dy_pixels * cm_per_pxl_h
        else: dy_cm = 0
        
        global_x += dx_cm
        global_y += dy_cm

        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        if prev_frame_time > 0:
            count_time += new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        updateLog(count_time,camera_height, dx_cm, dy_cm, global_x, global_y, int(fps))
        
        count_print+=1
        if count_print > 10:
            print("frame shape: {}".format(frame.shape))
            print("CM/pxl: {}x{}".format(round(cm_per_pxl_w,4),round(cm_per_pxl_h,4)))
            # print("Br = {:.2f}%, Dr = {:.2f}% ".format(br_percent,dr_percent))
            # print("Exposule: ", exposure)
            # print("Gain: ", gain)
            print("Pick point: ", len(good_new))
            print("Camera Height: ", camera_height)
            print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(dx_cm, dy_cm))
            print("X: {:.3f}, Y: {:.3f}".format(global_x, global_y))
            print("FPS: {}".format(int(fps)))
            print("FPS_cam: {}".format(int(fps_cam)))
            count_print = 0
    else:
        for i in range(10):
            print("Crash")
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    cv2.imshow("PiCam2", frame)
    old_gray = frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    del frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
##############################################################################

# Release memory
print("Close camera...")
# Release the Cameras
picam2.stop()
cv2.destroyAllWindows()
