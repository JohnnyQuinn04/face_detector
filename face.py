import cv2
import pyaudio
import numpy as np
import random

def audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)
    duration = 0.5  # seconds
    frequency = 840  # Hz
    samples_per_second = 44100
    t = np.linspace(0, duration, int(duration * samples_per_second), False)
    note = np.sin(frequency * t * 2 * np.pi)

    stream.write(note.astype(np.float32).tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        face_detected = True


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 225), 2)
        if face_detected:
            audio()
            print("Fire on yo dumbass")

    cv2.imshow('Face Detection', frame)

    # Exit the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
