from math import *
import threading
import concurrent.futures
import csv
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Queue
import multiprocessing as mp
from picamera2 import Picamera2
import time
import numpy as np
from matplotlib import pyplot as plt
import psutil
import cProfile

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

##################################################################################

######################### Camera function #########################

def doWork(st): #j=1 is left, j=2 is right
    grayL = st[0] 
    grayR = st[1]
    j = st[2]
    
    # Used for the filtered image
    if j == 1 :
        disp= stereo.compute(grayL,grayR)
    # Create another stereo for right this time
    if j == 2 :
        stereoR=cv2.ximgproc.createRightMatcher(stereo)
        disp= stereoR.compute(grayR,grayL)
    return disp

# Preprocessing function
def preprocess_frame(frame):
    width_crop = 320
    height_crop = 200
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
    return grayL,grayR

# Preprocessing function
def preprocess_optical_flow(frame):
    width_crop = 100
    height_crop = 100
    height, width = frame.shape
    # Take center image
    center_x = width // 2
    center_y = height // 2
    # Take position for crop
    start_x = center_x - (width_crop // 2)
    start_y = center_y - (height_crop // 2)
    # Take crop image
    frame = frame[start_y:start_y + height_crop, start_x:start_x + width_crop]
    return frame

# Frame capture thread
def capture_frames():
    global frame_buffer
    global fps_cam, fps_cam_buffer, key
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


# Define the task Stereo to be run by each process
def TakeStereo(id, t0stereo, grayL, grayR):
    fps_limit = 100
    # Filtering
    kernel= np.ones((3,3),np.uint8)
    # Compute the 2 images for the Depth_image
    st1 = (grayL,grayR,1 )
    st2 = (grayL,grayR,2 )
    disp , dispR = map(doWork, (st1,st2))
    dispL= disp
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)
    # Using the WLS filter
    cal_disparity = wls_filter.filter(dispL,grayL,None,dispR)
    cal_disparity= cv2.morphologyEx(cal_disparity,cv2.MORPH_CLOSE, kernel)
    # cal_disparity= ((cal_disparity.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
    cal_disparity = cv2.normalize(src=cal_disparity, dst=cal_disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    cal_disparity = np.uint8(cal_disparity)
    disparity_buffer = cal_disparity
    # Put result
    queue.put((disparity_buffer,grayL))

#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************
listPara = [i for i in range(10)]
paraCtl = {0:"block_size", 
        1:"min_disp", 
        2:"max_disp",
        3:"uniquenessRatio",
        4:"speckleWindowSize",
        5:"speckleRange",
        6:"disp12MaxDiff",
        7:"preFilterCap",
        8:"lmbda",
        9:"sigma"}
choosePara = 0

# Initialize StereoSGBM
block_size = 1
min_disp = 0
max_disp = 16*2
uniquenessRatio = 15
speckleWindowSize = 150
speckleRange = 5
disp12MaxDiff = 5
P1=8 * 3 * block_size * block_size
P2=32 * 3 * block_size * block_size
preFilterCap = 63
mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
lmbda = 80000
sigma = 1.8
num_disp = max_disp - min_disp

stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2,
        preFilterCap=preFilterCap,
        mode=mode,
    )
r_matcher = cv2.ximgproc.createRightMatcher(stereo)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

########### Optical Flow parameter #############
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
height_img = 800
div_crop = 1
crop_size_w = 320
crop_size_h = 200
focal_length = 0.14
pixel_dimension = 3 * 10**-4 
sensor_pxl_w = 400
sensor_pxl_h = 640
# cm_per_pxl_w_coef = (sensor_pxl_w * pixel_dimension / focal_length) / width_img 
# cm_per_pxl_h_coef = (sensor_pxl_h * pixel_dimension / focal_length) / height_img 
cm_per_pxl_w_coef = 1.625*10**-3
cm_per_pxl_h_coef = 1.625*10**-3
kalman = 0.7

#############################################################

im_display = None
filt_Color = None
fps = 0
fps_OpFlo = 0
fps_cam = 0
t0stereo = 0
t0OpFlo = 0
fps_stereo = 0
prev_frame_time = 0
new_frame_time = 0
count_time = 0
count_print = 0
dx_cm = 0
dy_cm = 0
global_x = 0
global_y = 0
old_gray = None
p0 = None
camera_height = 74

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
picam2 = Picamera2(0)

picam2.preview_configuration.main.size = (1600, 400)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 150.0
picam2.preview_configuration.align()

picam2.configure("preview")

picam2.start()

# Shared buffer for frames
frame = None
frame_buffer = None
frame_bufferL = None
fps_cam_buffer = None
buffer_lock = threading.Lock()

# Shared variable Process
disparity_buffer = None
fps_stereo_buffer = None
queue = Queue()
OpFlo_in_queue = Queue()
OpFlo_out_queue = Queue()

# Number of processes in the pool
max_processes = 2
futures = []
next_task_id = 0
limit_task_id = 1000

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Create a ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
    while True:
        with buffer_lock:
            if frame_buffer is not None:
                frame = frame_buffer
                frameL,frameR = preprocess_frame(frame)
                grayL = preprocess_optical_flow(frameL)
                fps_cam = 0.99 * fps_cam + 0.01*fps_cam_buffer
                frame_buffer = None
                fps_cam_buffer = None
                if old_gray is None:
                    old_gray = grayL
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                    continue
                if len(futures) <= max_processes*2:
                    if len(futures) <= max_processes:
                        new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frameL, frameR)
                        futures.append(new_future)
                        if next_task_id < limit_task_id:
                            next_task_id += 1
                        else:
                            next_task_id = 0
                    else:
                        if futures[0].done():
                            take_stereo_result = queue.get()
                            disparity_buffer = take_stereo_result[0] 
                            t1stereo = time.time()
                            fps_stereo_buffer = 1/ (t1stereo - t0stereo)
                            t0stereo = t1stereo
                            fps_stereo = 0.9* fps_stereo + 0.1*fps_stereo_buffer
                            futures.remove(futures[0])
                            new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frame)
                            futures.append(new_future)
                            if next_task_id < limit_task_id:
                                next_task_id += 1
                            else:
                                next_task_id = 0
            else:
                continue
        # Show result
        if disparity_buffer is not None:
            disparity = disparity_buffer
            disparity_buffer = None
            filt_Color = cv2.applyColorMap(disparity,cv2.COLORMAP_JET)
        # if filt_Color is not None:
        #     cv2.imshow('Filtered Color Depth',filt_Color)
        #     #cv2.imshow('Depth',disparity)  
        #     cv2.imshow('Both Images', frameL) 
        #     new_frame_time = time.time()
        #     fps = 0.9 * fps + 0.1 / (new_frame_time - prev_frame_time)
        #     prev_frame_time = new_frame_time
        
        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, grayL, p0, None, **lk_params)
        except:
            for i in range(10):
                print("False tracking!!!!!!!!!!!!")
            old_gray = grayL
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            global_x += dx_cm
            global_y += dy_cm
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # # Show detail of flow if need
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
            
            if abs(dx_pixels * cm_per_pxl_w) >= cm_per_pxl_w/7: 
                dx_cm = (1 - kalman) * dx_cm + kalman * dx_pixels * cm_per_pxl_w
            else: dx_cm = 0
            if abs(dy_pixels * cm_per_pxl_h) >= cm_per_pxl_h/7: 
                dy_cm = (1 - kalman) * dy_cm + kalman * dy_pixels * cm_per_pxl_h
            else: dy_cm = 0
            
            global_x += dx_cm
            global_y += dy_cm
            t1OpFlo = time.time()
            fps_OpFlo = 0.9* fps_OpFlo + 0.1/(t1OpFlo - t0OpFlo)
            # if t0OpFlo > 0:
            #     count_time += t1OpFlo - t0OpFlo
            t0OpFlo = t1OpFlo
            # count_print+=1
            # if count_print > 10:
            #     # print("frame shape: ",format(frame.shape))
            #     # print("CM/pxl: ", round(cm_per_pxl_w/7,3))
            #     # print("Br = {:.2f}%, Dr = {:.2f}% ".format(br_percent,dr_percent))
            #     # print("Exposule: ", exposure)
            #     # print("Gain: ", gain)
            #     # print("Pick point: ", len(good_new))
            #     print("Camera Height: ", camera_height)
            #     print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(dx_cm, dy_cm))
            #     print("X: {:.3f}, Y: {:.3f}".format(global_x, global_y))
            #     print("FPS: {}".format(int(fps)))
            #     print("FPS_cam: {}".format(int(fps_cam)))
            #     count_print = 0
        else:
            for i in range(10):
                print("Crash")
            old_gray = grayL
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            continue
        
        # Print results
        count_print+=1
        if count_print > 1:
            # print("{}= {}".format(paraCtl[0], block_size))
            # print("{}= {}".format(paraCtl[1], min_disp))
            # print("{}= {}".format(paraCtl[2], max_disp))
            # print("{}= {}".format(paraCtl[3], uniquenessRatio))
            # print("{}= {}".format(paraCtl[4], speckleWindowSize))
            # print("{}= {}".format(paraCtl[5], speckleRange))
            # print("{}= {}".format(paraCtl[6], disp12MaxDiff))
            # print("{}= {}".format(paraCtl[7], preFilterCap))
            # print("{}= {}".format(paraCtl[8], lmbda))
            # print("{}= {}".format(paraCtl[9], sigma))
            # print("Param control: {}".format(paraCtl[choosePara]))
            print("__________________________________________________")
            # print("threading.activeCount():",threading.activeCount())
            # print("threading.currentThread():",threading.currentThread())
            # print("threading.enumerate():",threading.enumerate())
            print("Camera Height: ", camera_height)
            print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(dx_cm, dy_cm))
            print("X: {:.3f}, Y: {:.3f}".format(global_x, global_y))
            #print("FPS: {}".format(int(fps)))
            print("FPS_cam: {}".format(int(fps_cam)))
            print("FPS_stereo: {}".format(int(fps_stereo)))
            print("FPS_OpticalFlow: {}".format(int(fps_OpFlo)))
        key = cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    executor.shutdown(wait=False,cancel_futures=True)

#ser.close()
# Release memory
del frame
print("Close camera...")
# Release the Cameras
picam2.stop()
cv2.destroyAllWindows()