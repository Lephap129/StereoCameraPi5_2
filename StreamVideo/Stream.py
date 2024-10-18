from flask import Flask, render_template, Response
import cv2
from picamera2 import Picamera2
import multiprocessing as mp
app = Flask(__name__)

# Initialize the Picamera2 (or you can use any other camera)
picam2 = Picamera2(0)
picam2.preview_configuration.main.size = (1600, 400)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 120.0
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def gen_frames():
    """Generate frame by frame from the camera and encode it to JPEG"""
    while True:
        frame = picam2.capture_array()  # Capture frame from camera
        if frame is None:
            continue
        # Processing if needed
        # Encode the frame into JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming homepage."""
    return render_template('index.html')

def main():
    app.run(host='0.0.0.0', port=5000)

process1 = mp.Process(name = "Process-1", target=main)
process1.start()
a = 0
while True:
    if a < 100: 
        print(a)
        a += 1
