from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()

@app.get("/")
async def get_camera():
    cap = cv2.VideoCapture(1)

    async def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')