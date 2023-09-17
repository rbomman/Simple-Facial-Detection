import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
video.set(3,1920) # The width of the screen
video.set(4,1080) # The height of the screen

while True:
    ret, img = video.read() #get a frame from the video
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale for the model 
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    ) #find faces in the video frame
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw a circle from bottom left (x,y) to top right (x+h, y+h) of detected face
    
    cv2.imshow('video',img) #display the frames as a video
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    
video.release()
cv2.destroyAllWindows()
