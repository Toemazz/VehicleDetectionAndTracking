import cv2
import numpy as np

frame = cv2.imread('output/frame1.png')

dims = frame.shape[:2]
cv2.putText(frame, 'WARNING', (dims[1]-300, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(frame, 'WARNING', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(frame, 'SAFE', (int(dims[1]/2)-50, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('result', frame)
cv2.waitKey(0)

frame2 = np.hstack([frame, frame])
frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
dims2 = frame2.shape[:2]
cv2.putText(frame2, 'SAFE', (int(dims2[1]/2)-40, 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('result2', frame2)
cv2.waitKey(0)
