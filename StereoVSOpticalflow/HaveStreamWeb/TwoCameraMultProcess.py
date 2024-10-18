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
from PIL import Image
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi import FastAPI, Response
import io

import uvicorn.logging

##################################################################################
# region: Record data
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
# endregion
##################################################################################

##################################################################################
# region: Camera function
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
    width_crop = 300
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
    grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)
    # grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    # grayL = cv2.medianBlur(grayL, 5)
    # grayR = cv2.GaussianBlur(grayR, (5, 5), 0)
    # grayR = cv2.medianBlur(grayR, 5)
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
# Take distance
def take_distance(disp):
    width_crop = 100
    height_crop = 100
    height, width = disp.shape
    # Take center image
    center_x = width // 2
    center_y = height // 2
    # Take position for crop
    start_x = center_x - (width_crop // 2)
    start_y = center_y - (height_crop // 2)
    # Take crop image
    disp = disp[start_y:start_y + height_crop, start_x:start_x + width_crop]
    avg = np.mean(disp)
    # try:
    #     Distance= np.around(70275*avg**(-0.976),decimals=2)
    # except:
    #     Distance = 999999.999
    Distance = 77
    return Distance
# endregion
##################################################################################

# region: Frame capture thread
##################################################################################
def capture_frames():
    global frame_buffer
    global fps_cam, fps_cam_buffer, key
    t0cam = 0
    fps_limit = 60 
    while True:
        # Guard CPU performance
        if fps_limit > 20 and psutil.cpu_percent() > 90:
            fps_limit -= 10
        elif fps_limit < 60 and psutil.cpu_percent() < 60:
            fps_limit += 10  
        if psutil.cpu_percent() > 90:
            time.sleep(0.01)
        if (fps_cam > fps_limit):
            time.sleep(round(1/fps_limit - 1/fps_cam, 3))
        # Take frame
        frame = picam2.capture_array()
        with buffer_lock:
            frame_buffer = frame
            t1cam = time.time()
            fps_cam_buffer = 1/(t1cam -t0cam)
            t0cam = t1cam
##################################################################################

# region: Task Stereo process
##################################################################################
def TakeStereo(id, t0stereo, grayL, grayR, fps):
    fps_limit = 10
    if (fps > fps_limit):
        time.sleep(round(1/fps_limit - 1/fps, 3))
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
    height = take_distance(disp)
    # Put result
    queue.put((disparity_buffer,grayL, height))
##################################################################################

# region: Task Optical Flow process  
##################################################################################
def OpticalFlow(id, old_gray, p0, camera_height, 
                frame, dx_cm, dy_cm, fps):
    fps_limit = 80
    if (fps > fps_limit):
        time.sleep(round(1/fps_limit - 1/fps, 3))
    camera_height = 77
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
    cm_per_pxl_w_coef = 1.9125*10**-3
    cm_per_pxl_h_coef = 1.9125*10**-3

    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame, p0, None, **lk_params)
    except:
        for i in range(20):
            print("False tracking!!!!!!!!!!!!")
        OpFlo_out_queue.put((dx_cm, dy_cm, None, None))
        return id

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
            dx_cm = dx_pixels * cm_per_pxl_w
        else: dx_cm = 0
        if abs(dy_pixels * cm_per_pxl_h) >= cm_per_pxl_h/7: 
            dy_cm = dy_pixels * cm_per_pxl_h
        else: dy_cm = 0

        OpFlo_out_queue.put((dx_cm, dy_cm, good_new, good_old)) 
        return id
    else:
        for i in range(20):
            print("Crash")
        dx_cm = 0
        dy_cm = 0
        OpFlo_out_queue.put((dx_cm, dy_cm, None, None))
        return id

##################################################################################


#*********************************************************************************
# region: StereoVision parameter
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
# endregion
#*********************************************************************************

##################################################################################
# region: Other parameter
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
camera_height = 0
kalman = 0.7
# endregion
##################################################################################

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

##################################################################################
# region: Initial program
# Debug
str_error = ''
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
OpFlo_out_queue = Queue()

# Number of processes in the pool
max_processes = 5
stereo_futures = []
optical_flow_futures = []
next_task_id = 0
limit_task_id = 1000

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()
# endregion
##################################################################################
app = FastAPI()
def generate_frames():
    global im_display, filt_Color, fps, fps_OpFlo, fps_cam, t0stereo, t0OpFlo, fps_stereo, prev_frame_time, new_frame_time, count_time, count_print
    global dx_cm, dy_cm, global_x, global_y, old_gray, p0, camera_height, kalman
    global block_size, min_disp, max_disp, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, preFilterCap, lmbda, sigma, num_disp
    global frame, frame_buffer, frame_bufferL, fps_cam_buffer, buffer_lock
    global disparity_buffer, fps_stereo_buffer, queue, OpFlo_out_queue 
    global max_processes, stereo_futures, optical_flow_futures, next_task_id, limit_task_id
    P1=8 * 3 * block_size * block_size
    P2=32 * 3 * block_size * block_size
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    #*********************************************************************************
    # region: StereoVision
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
    # endregion
    #*********************************************************************************
    
    ##################################################################################
    # region: Optical Flow parameter
    # Lucas-Kanade parameters
    lk_params = dict(winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Detect feature points parameters
    feature_params = dict(maxCorners=20,
                        qualityLevel=0.3,
                        minDistance=7,
                        blockSize=7)
    # endregion
    ##################################################################################

    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        while True:
            str_error = ''
            with buffer_lock:
                if frame_buffer is not None:
                    frame = frame_buffer
                    frameL,frameR = preprocess_frame(frame)
                    grayL = preprocess_optical_flow(frameL)
                    # grayL = frameLs   
                    fps_cam = 0.99 * fps_cam + 0.01*fps_cam_buffer
                    frame_buffer = None
                    fps_cam_buffer = None
                    if old_gray is None:
                        old_gray = grayL
                        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                        continue
                    number_task = len(stereo_futures)+len(optical_flow_futures)
                    if  number_task <= max_processes*4:
                        if number_task <= max_processes:
                            if number_task%5 == 0:
                                new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frameL, frameR, fps_stereo)
                                stereo_futures.append(new_future)
                            else:
                                OpFloFrame = preprocess_optical_flow(frameL)
                                # OpFloFrame = frameL
                                new_future = executor.submit(OpticalFlow, next_task_id, 
                                                            old_gray, p0, camera_height, 
                                                            OpFloFrame, dx_cm, dy_cm, fps_OpFlo)
                                old_gray = OpFloFrame
                                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                                optical_flow_futures.append(new_future)
                            if next_task_id < limit_task_id:
                                next_task_id += 1
                            else:
                                next_task_id = 0
                        else:
                            if stereo_futures[0].done():
                                take_stereo_result = queue.get()
                                disparity_buffer = take_stereo_result[0]
                                if camera_height == 0 or take_stereo_result[2] == 999999.999:
                                    camera_height = take_stereo_result[2] 
                                else:
                                    if camera_height == 999999.999:
                                        camera_height = 0
                                    camera_height = 0.9*camera_height + 0.1*take_stereo_result[2] 
                                t1stereo = time.time()
                                fps_stereo_buffer = 1/ (t1stereo - t0stereo)
                                t0stereo = t1stereo
                                fps_stereo = 0.99* fps_stereo + 0.01*fps_stereo_buffer
                                stereo_futures.remove(stereo_futures[0])
                                new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frameL, frameR, fps_stereo)
                                stereo_futures.append(new_future)
                                if next_task_id < limit_task_id:
                                    next_task_id += 1
                                else:
                                    next_task_id = 0
                            elif optical_flow_futures[0].done():
                                take_OpFlo_result = OpFlo_out_queue.get()
                                if take_OpFlo_result[0] != 0:
                                    dx_cm = (1 - kalman) * dx_cm + kalman * take_OpFlo_result[0]
                                    global_x += dx_cm
                                else:
                                    dx_cm = (1 - kalman) * dx_cm + kalman * take_OpFlo_result[0]
                                if take_OpFlo_result[1] != 0:
                                    dy_cm = (1 - kalman) * dy_cm + kalman * take_OpFlo_result[1]
                                    global_y += dy_cm
                                else: 
                                    dy_cm = (1 - kalman) * dy_cm + kalman * take_OpFlo_result[1]
                                                                   
                                t1OpFlo = time.time()
                                fps_OpFlo = 0.99* fps_OpFlo + 0.01/(t1OpFlo - t0OpFlo)
                                if t0OpFlo > 0:
                                    count_time += t1OpFlo - t0OpFlo
                                t0OpFlo = t1OpFlo
                                optical_flow_futures.remove(optical_flow_futures[0])
                                OpFloFrame = preprocess_optical_flow(frameL)
                                new_future = executor.submit(OpticalFlow, next_task_id, 
                                                            old_gray, p0, camera_height, 
                                                            OpFloFrame, dx_cm, dy_cm, fps_OpFlo)
                                updateLog(count_time, camera_height, dx_cm, dy_cm, global_x, global_y, int(fps_OpFlo))
                                old_gray = OpFloFrame
                                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                                optical_flow_futures.append(new_future)
                                if next_task_id < limit_task_id:
                                    next_task_id += 1
                                else:
                                    next_task_id = 0
                                    
            # Show result
            if disparity_buffer is not None:
                disparity = disparity_buffer
                disparity_buffer = None
                filt_Color = cv2.applyColorMap(disparity,cv2.COLORMAP_JET)
            if filt_Color is not None:
                # cv2.imshow('Filtered Color Depth',filt_Color)
                # cv2.imshow('Depth',disparity)  
                # cv2.imshow('Both Images', frameL)
                img1 = Image.fromarray(frameL)
                img_io1 = io.BytesIO()
                img1.save(img_io1, 'JPEG')
                img_io1.seek(0)
                
                img2 = Image.fromarray(filt_Color)
                img_io2 = io.BytesIO()
                img2.save(img_io2, 'JPEG')
                img_io2.seek(0)
                # # Show detail of flow if need
                # sFrame = OpFloFrame.copy()
                # if good_new is not None:
                #     for i, (new, old) in enumerate(zip(good_new, good_old)):
                #         a, b = new.ravel()
                #         c, d = old.ravel()
                #         cv2.line(sFrame, (a, b), (c, d), 255, 1)
                #         cv2.circle(sFrame, (a, b), 3, 255, -1)
                #     good_new = None
                #     good_old = None
                img3 = Image.fromarray(OpFloFrame)
                img_io3 = io.BytesIO()
                img3.save(img_io3, 'JPEG')
                img_io3.seek(0)
                
                 # Combine images A, B, and C side by side
                combined_width = img1.width * 3
                combined_height = max(img1.height, img2.height, img3.height)
                combined_image = Image.new('RGB', (combined_width, combined_height))

                combined_image.paste(img1, (0, 0))  # Paste Image 1
                combined_image.paste(img2.convert("RGB"), (img1.width, 0))  # Paste Image 2
                combined_image.paste(img3.convert("RGB"), (img1.width * 2, 0))  # Paste Image 3

                # Save the combined image to a byte stream
                img_io_combined = io.BytesIO()
                combined_image.save(img_io_combined, 'JPEG')
                img_io_combined.seek(0)

                # Yield the JPEG image as part of the MJPEG stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_io_combined.read() + b'\r\n')  
                new_frame_time = time.time()
                fps = 0.99*fps + 0.01 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
            
            # # Print results
            # count_print+=1
            # if count_print > fps/2:
            #     count_print = 0
            #     # print("{}= {}".format(paraCtl[0], block_size))
            #     # print("{}= {}".format(paraCtl[1], min_disp))
            #     # print("{}= {}".format(paraCtl[2], max_disp))
            #     # print("{}= {}".format(paraCtl[3], uniquenessRatio))
            #     # print("{}= {}".format(paraCtl[4], speckleWindowSize))
            #     # print("{}= {}".format(paraCtl[5], speckleRange))
            #     # print("{}= {}".format(paraCtl[6], disp12MaxDiff))
            #     # print("{}= {}".format(paraCtl[7], preFilterCap))
            #     # print("{}= {}".format(paraCtl[8], lmbda))
            #     # print("{}= {}".format(paraCtl[9], sigma))
            #     # print("Param control: {}".format(paraCtl[choosePara]))
            #     print("__________________________________________________")
            #     # print("threading.activeCount():",threading.activeCount())
            #     # print("threading.currentThread():",threading.currentThread())
            #     # print("threading.enumerate():",threading.enumerate())
            #     print("Optical Flow future size:", len(optical_flow_futures))
            #     print("Stereo future size:", len(stereo_futures))
            #     print("Camera Height: ", camera_height)
            #     print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(dx_cm, dy_cm))
            #     print("X: {:.3f}, Y: {:.3f}".format(global_x, global_y))
            #     print("FPS: {}".format(int(fps)))
            #     print("FPS_cam: {}".format(int(fps_cam)))
            #     print("FPS_stereo: {}".format(int(fps_stereo)))
            #     print("FPS_OpticalFlow: {}".format(int(fps_OpFlo)))

    #ser.close()
    # Release memory
    del frame
    print("Close camera...")
    # Release the Cameras
    picam2.stop()
    cv2.destroyAllWindows()
        
@app.get("/video-stream")
def video_stream():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/info")
def get_info():
    # Return the FPS and parameter control values as JSON
    return JSONResponse(content={
        "optical_flow_futures": len(optical_flow_futures),
        "stereo_futures": len(stereo_futures),
        "camera_height": round(camera_height,2),
        "dx_cm": round(dx_cm, 3),
        "dy_cm": round(dy_cm, 3),
        "global_x": round(global_x, 3),
        "global_y": round(global_y, 3),
        "FPS": int(fps),
        "FPS_cam": int(fps_cam),
        "FPS_stereo": int(fps_stereo),
        "FPS_OpticalFlow": int(fps_OpFlo),
    })

@app.post("/reset")
def reset_parameters():
    global fps_OpticalFlow, im_display, filt_Color, fps, fps_OpFlo, fps_cam, t0stereo, t0OpFlo, fps_stereo, prev_frame_time, new_frame_time, count_time, count_print, dx_cm, dy_cm, global_x, global_y, old_gray, p0, camera_height

    # Reset all parameters to default values
    fps_OpticalFlow = 0
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
    camera_height = 0

    return {"message": "Parameters reset to default"}

@app.get("/")
def index():
    # Serve the HTML file from the static directory
    return FileResponse(os.path.join("templates", "index2.html"))
import logging
# Set up custom logging configuration
def custom_logging():
    # Get Uvicorn's logger
    uvicorn_logger = logging.getLogger("uvicorn")
    
    # Set the level of the logger to INFO (or any other level you need)
    uvicorn_logger.setLevel(logging.INFO)

    # Create a custom handler for stdout
    console_handler = logging.StreamHandler()

    # Set the logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the Uvicorn logger
    uvicorn_logger.addHandler(console_handler)

    # Optionally remove Uvicorn's default handlers if necessary
    uvicorn_logger.handlers.clear()
    uvicorn_logger.addHandler(console_handler)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info", access_log=False)
    