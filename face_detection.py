import cv2

# 1. Load the AI checklists for BOTH faces and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 2. Turn on your webcam
cap = cv2.VideoCapture(0)

while True:
    # 3. Read the frame and convert to black and white
    success, frame = cap.read()
    if not success:
        print("Camera not working!")
        break
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Search for faces in the whole screen
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # 5. Loop through every face found
    for (x, y, w, h) in faces:
        # Draw a blue box around the face (OpenCV uses BGR color format: Blue, Green, Red)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 6. Create the "Region of Interest" (ROI)
        # This acts like a digital crop of just the face area
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # 7. Search for eyes ONLY inside the cropped gray face area
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        
        # 8. Draw a green box around any eyes found inside the face
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # 9. Show the video on screen
    cv2.imshow('Advanced AI Face & Eye Detector', frame)

    # 10. Wait for the user to press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 11. Turn off the camera and close windows
cap.release()
cv2.destroyAllWindows()
