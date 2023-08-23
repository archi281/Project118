import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier( cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/archi/OneDrive/Desktop/WhiteHatJr/Pro C118 Prj/PRO-106-ProjectTemplate-main/walking.avi')

# Loop once video is successfully loaded
while (True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_classifier.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(25) == 32: 
        break

cap.release()

cv2.destroyAllWindows()
