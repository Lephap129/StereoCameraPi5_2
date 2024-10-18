import cv2
import time
from picamera2 import Picamera2
import threading

######################### Camera setting function #########################
def preprocess_frame(frame):
    # Split left and right images
    height, width, _ = frame.shape
    frameL = frame[:, :width // 2]
    frameR = frame[:, width // 2:]
    
    return frameL, frameR

# Frame capture thread
def capture_frames():
    global frame_buffer, buffer_lock

    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Store frame in buffer
        with buffer_lock:
            frame_buffer = frame

##################################################################################

######################### Initial function and parameter #########################

# Initialize the camera
picam2 = Picamera2(0)

# Set camera resolution and format
picam2.preview_configuration.main.size = (1600, 400)  # Width is 1600 for two cameras (800 each)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 150.0
picam2.preview_configuration.align()

picam2.configure("preview")
picam2.start()

# Shared buffer for frames
frame_buffer = None
buffer_lock = threading.Lock()

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

##################################################################################

######################### Main processing loop ###############################
image_count = 0
while True:
    with buffer_lock:
        if frame_buffer is not None:
            frame = frame_buffer
            frameL, frameR = preprocess_frame(frame)

            # Save left and right frames as PNG images
            cv2.imwrite(f'left_image_{image_count}.png', frameL)
            cv2.imwrite(f'right_image_{image_count}.png', frameR)
            print(f"Saved images left_image_{image_count}.png and right_image_{image_count}.png")

            image_count += 1
            frame_buffer = None

    time.sleep(1)  # Delay to capture images at intervals, adjust as needed

    if image_count >= 10:  # Capture 10 image pairs and then stop
        break

##################################################################################

# Release memory
print("Close camera...")
picam2.stop()
cv2.destroyAllWindows()
