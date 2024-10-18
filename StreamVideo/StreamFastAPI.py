from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
import os
from picamera2 import Picamera2
import io
import time
from PIL import Image

app = FastAPI()

# Initialize Picamera2
picam2 = Picamera2()

# Configure camera settings
picam2.preview_configuration.main.size = (1600, 400)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 120.0
picam2.configure("preview")
picam2.start()

fps = 0
fps_cam = 0
fps_stereo = 0
choosePara = 0
paraCtl = {0: "block_size", 1: "min_disp", 2: "max_disp", 3: "uniquenessRatio", 4: "speckleWindowSize",
           5: "speckleRange", 6: "disp12MaxDiff", 7: "preFilterCap", 8: "lmbda", 9: "sigma"}

def generate_frames():
    global fps
    prev_frame_time = time.time()
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()
        
        # Convert frame (NumPy array) to JPEG
        img = Image.fromarray(frame)
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Yield the JPEG image as part of the MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_io.read() + b'\r\n')

        # Control frame rate by adding a small delay (optional)
        time.sleep(0.01)  # Adjust delay to control FPS

@app.get("/video-stream")
def video_stream():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/info")
def get_info():
    # Return the FPS and parameter control values as JSON
    return JSONResponse(content={
        "FPS": round(fps, 2),
        "FPS_cam": round(fps * 0.9, 2),  # Simulated camera FPS
        "FPS_stereo": round(fps * 0.8, 2),  # Simulated stereo FPS
        "Param_control": paraCtl[choosePara]
    })

@app.get("/")
def index():
    # Serve the HTML file from the static directory
    return FileResponse(os.path.join("templates", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
