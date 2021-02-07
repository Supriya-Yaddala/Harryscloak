import cv2
import numpy as np

cap=cv2.VideoCapture(0)
while True:
    ret,init_frame=cap.read()

    if ret==True:
        break
while True:
    ret,frame=cap.read()

    grayvidframe=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
   # cv2.imshow('initframe',init_frame)

    lower = np.array([150, 100, 100])
    upper = np.array([180, 255, 255])
    kernel=np.ones((3,3),np.uint8)
    mask=cv2.inRange(grayvidframe,lower,upper)

    mask=cv2.medianBlur(mask,3)
    #cv2.imshow('mask', mask)

    mask_inv = cv2.bitwise_not(mask)
    #cv2.imshow('mask1', mask_inv)
    mask=cv2.dilate(mask,kernel,5)


    #getting frame background
    b=frame[:,:,0]
    g=frame[:,:,1]
    r=frame[:,:,2]
    b=cv2.bitwise_and(b,mask_inv)
    g=cv2.bitwise_and(g,mask_inv)
    r=cv2.bitwise_and(r,mask_inv)
    frame_bg=cv2.merge((b,g,r))

    #getting foreground
    b = init_frame[:, :, 0]
    g = init_frame[:, :, 1]
    r = init_frame[:, :, 2]
    b=cv2.bitwise_and(b,b,mask=mask)
    g = cv2.bitwise_and(g,g,mask=mask)
    r = cv2.bitwise_and(r,r,mask=mask)
    frame_fg = cv2.merge((b, g, r))
    harryscloak=cv2.bitwise_or(frame_fg,frame_bg)

    cv2.imshow('mask2',harryscloak)


    if cv2.waitKey(1)==27:
        break


cv2.destroyAllWindows()
cap.release()
