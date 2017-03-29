#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: SAVITHRU M LOKANATH

import cv2
import sys

def draw(faces,frame):

	for (x, y, w, h) in faces:
	        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():

	cascPath = sys.argv[1]
	faceCascade = cv2.CascadeClassifier(cascPath)

	video_capture = cv2.VideoCapture(0)

	while True:
    		ret, frame = video_capture.read()

    		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    		faces = faceCascade.detectMultiScale(
       		    gray,
                    scaleFactor=1.1,
        	    minNeighbors=10,
        	    minSize=(60, 60),
        	    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    	        )
    	
    		draw(faces,frame)

		cv2.imshow('FaceDetect', frame)

        	if cv2.waitKey(1) & 0xFF == ord('q'):
                	break

	video_capture.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	main()
