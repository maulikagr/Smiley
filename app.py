import numpy as np
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(25, 25))

                for i in smile:
                    if len(smile) > 1:
                        cv2.putText(frame, "Smiling", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
                                    cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Not Smiling", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                                    cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port = 5000)
