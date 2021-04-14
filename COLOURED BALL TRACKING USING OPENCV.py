import cv2
from imutils.video import VideoStream
import numpy as np
import imutils
from collections import deque






# find the upper and lower limit of colour of your ball 
greenLower = (24, 68, 96)
greenUpper = (88, 255, 255)
pts = deque(maxlen = 32)
# used deque as it can add or remove from either end 

# start capturing frame by frame
cap = cv2.VideoCapture(0)
cv2.namedWindow('Ball tracker')
# window to display the output

# each frame will pass thru this

while True:
    ret, frame = cap.read()
    if not ret:
        break
     # if the cam reads a frame only then it will move ahead
     
    frame = cv2.flip( frame, 1 )
    #flipped frame to avoid mirroring
    frame = imutils.resize(frame,width = 600)
    blurred = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    # Blurred the frame and converted it HSV
    
    mask = cv2.inRange(hsv,greenLower,greenUpper)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    # applied mask for extracting the colour from the background
    
    # finding contours that the coloured ball make
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    # only moving ahead if atleast one contour is found
    if len(cnts) > 0:
        c = max(cnts, key = cv2.contourArea)
        # taking the max values
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        # creating a enclosing circle around those contours
        m = cv2.moments(c)
        center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        # finding centeroid using moments
        
        if radius > 10:
            cv2.circle(frame,(int(x), int(y)), int(radius), (0,0,255), 3)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # drawing the circle using the data we extracted
        
    pts.appendleft(center)
    # storing the center values in deque from left
    
    # drawing line between 2 consecutive points also tweaked the thickness to get min thickness at the end of trail
    for i in range (1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thk = int(np.sqrt(32/float(i + 1))*2.5)
        cv2.line(frame, pts[i-1], pts[i], (0,0,255), thk )
    
    
    # display the final processed frame on the window we created at start
    cv2.imshow('Ball tracker',frame)
    
    # stop when user presses esc key
    key = cv2.waitKey(1)
    if key == 27:
        break
        
# release and destroy all windows and terminate the program        
cap.release()
cv2.destroyAllWindows()
