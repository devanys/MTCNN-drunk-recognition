import cv2
import numpy as np
from mtcnn import MTCNN

def detect_red_face(frame, face):
    x, y, width, height = face['box']
    roi_color = frame[y:y+height, x:x+width]
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    num_red_pixels = cv2.countNonZero(mask)
    
    area = width * height
    red_ratio = num_red_pixels / area
    
    if red_ratio > 0.03:
        return True
    else:
        return False

detector = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(rgb_frame)
    
    if faces:
        for face in faces:
            if detect_red_face(frame, face):
                label = "Drunk"
                color = (0, 0, 255) 
            else:
                label = "Sober"
                color = (0, 255, 0)  
            
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow('Video Drunk Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
