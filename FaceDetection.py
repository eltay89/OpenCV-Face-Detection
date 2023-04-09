import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a video capture device (in this case, the default camera)
cap = cv2.VideoCapture(0)

# Start an infinite loop that captures frames from the camera and performs face detection on each frame
while True:
    # Capture a frame from the video capture device
    ret, frame = cap.read()

    # Convert the captured frame from BGR color space to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Draw a rectangle around each detected face on the original (color) frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the original (color) frame with the detected faces overlaid on it
    cv2.imshow('Webcam', frame)

    # Wait for a key press for 20 milliseconds, and check if the pressed key is 'q'. If so, break out of the infinite loop.
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
