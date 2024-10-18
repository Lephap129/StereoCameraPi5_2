from math import *
import threading
import concurrent.futures
import csv
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Queue
from picamera2 import Picamera2
import time
import numpy as np
from matplotlib import pyplot as plt
import psutil
from PIL import Image
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi import FastAPI, Response
import io
import signal
import sys

######################### Record data ########################## 
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()


def updateLog(t,h,dx,dy,X,Y,FPS, exposure, br_percent, dr_percent):
    list_append = [{'t':'{:.04f}'.format(t),'h':'{:.02f}'.format(h),
                    'dx': '{:.02f}'.format(dx), 'dy': '{:.02f}'.format(dy), 
                    'X': '{:.02f}'.format(X), 'Y': '{:.02f}'.format(Y), 
                    'FPS': '{}'.format(FPS), 'exposure': '{}'.format(exposure),
                    'br': '{}'.format(br_percent), 'dr': '{}'.format(dr_percent)}]
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
def TakeStereo(id, t0stereo, frame):
    fps_limit = 100
    grayL, grayR = preprocess_frame(frame)
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


im_display = None
fps = 0
fps_cam = 0
t0stereo = 0
fps_stereo = 0
prev_frame_time = 0
new_frame_time = 0
count_print = 0

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
picam2 = Picamera2(0)

picam2.preview_configuration.main.size = (1600, 400)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 120.0
picam2.preview_configuration.align()

picam2.configure("preview")

picam2.start()

# Shared buffer for frames
frame_buffer = None
frame_bufferL = None
fps_cam_buffer = None
buffer_lock = threading.Lock()

# Shared variable Process
disparity_buffer = None
fps_stereo_buffer = None
queue = Queue()

# Number of processes in the pool
max_processes = 2
futures = []
next_task_id = 0
limit_task_id = 1000
# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

app = FastAPI()
def generate_frames():
    global im_display, fps, fps_cam, t0stereo, fps_stereo, prev_frame_time, new_frame_time, count_print, key
    global block_size, min_disp, max_disp, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, preFilterCap, lmbda, sigma, num_disp
    global frame_buffer, frame_bufferL, fps_cam_buffer, buffer_lock, disparity_buffer, fps_stereo_buffer, queue, max_processes, futures, next_task_id, limit_task_id, executor
    P1=8 * 3 * block_size * block_size
    P2=32 * 3 * block_size * block_size
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
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
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        while True:
            with buffer_lock:
                if frame_buffer is not None:
                    frame = frame_buffer
                    cv2.imwrite("stream.jpg",frame)
                    fps_cam = 0.99 * fps_cam + 0.01*fps_cam_buffer
                    frame_buffer = None
                    fps_cam_buffer = None
                    if len(futures) <= max_processes*2:
                        if len(futures) <= max_processes:
                            new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frame)
                            futures.append(new_future)
                            if next_task_id < limit_task_id:
                                next_task_id += 1
                            else:
                                next_task_id = 0
                        else:
                            if futures[0].done():
                                # print("1.2.1")
                                take_stereo_result = queue.get()
                                disparity_buffer = take_stereo_result[0] 
                                frame_bufferL = take_stereo_result[1]
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
            # Show result
            if disparity_buffer is not None:
                disparity = disparity_buffer
                frameL = frame_bufferL
                disparity_buffer = None
                frame_bufferL = None
                filt_Color= cv2.applyColorMap(disparity,cv2.COLORMAP_JET)
                # cv2.imshow('Filtered Color Depth',filt_Color)
                # cv2.imshow('Depth',disparity)  
                # cv2.imshow('Both Images', frameL)
                # Convert frame (NumPy array) to JPEG
                img1 = Image.fromarray(frameL)
                img_io1 = io.BytesIO()
                img1.save(img_io1, 'JPEG')
                img_io1.seek(0)
                
                img2 = Image.fromarray(filt_Color)
                img_io2 = io.BytesIO()
                img2.save(img_io2, 'JPEG')
                img_io2.seek(0)
                
                img3 = Image.fromarray(disparity)
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
                fps = 0.9 * fps + 0.1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                # Print results
                count_print+=1
                if count_print > fps/2:
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
                    print("__________________________________________________")
                    # print("threading.activeCount():",threading.activeCount())
                    # print("threading.currentThread():",threading.currentThread())
                    # print("threading.enumerate():",threading.enumerate())
                    print("FPS: {}".format(int(fps)))
                    print("FPS_cam: {}".format(int(fps_cam)))
                    print("FPS_stereo: {}".format(int(fps_stereo)))
                    print("Param control: {}".format(paraCtl[choosePara]))
        
@app.get("/video-stream")
def video_stream():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/info")
def get_info():
    # Return the FPS and parameter control values as JSON
    return JSONResponse(content={
        "FPS": round(fps, 2),
        "FPS_cam": round(fps_cam, 2),
        "FPS_stereo": round(fps_stereo, 2),
    })

@app.get("/")
def index():
    # Serve the HTML file from the static directory
    return FileResponse(os.path.join("templates", "index.html"))

# Gracefully stop the program and camera when Ctrl+C is pressed
def handle_exit(signum, frame):
    executor.shutdown(wait=False,cancel_futures=True)
    print("Exiting program...")
    picam2.stop()  # Stop the camera
    sys.exit(0)

# Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except (KeyboardInterrupt, SystemExit):
        handle_exit(None, None)