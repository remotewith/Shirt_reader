#Pose-Estimation
#Mediapipe takes frames as in form of RGB format


import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pytesseract


mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


#cap=cv2.VideoCapture('http://[2409:4071:4e9c:211e::6f]:8080/video')
cap=cv2.VideoCapture(0)
_,prev=cap.read()
prev=cv2.flip(prev,1)
_,new=cap.read()
new=cv2.flip(new,1)
t=True
lmList1=[]
a=0
l1,l2=[],[]
while True:
    lmList=[]
    
    _,img=cap.read()
    

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            
            h,w,c=img.shape
            T=img.shape
            
            cx,cy=int(lm.x*w),int(lm.y*h)
            #print(id,cx,cy)
            lmList.append([id,cx,cy])
            lmList1.append([id,cx,cy])
            #print(lmList)
            if len(lmList)!=0 and id==32:

                x1,y1=lmList[12][1],lmList[12][2]
                x2,y2=lmList[23][1],lmList[23][2]

                pts1=np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])

                pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])

                matrix=cv2.getPerspectiveTransform(pts1,pts2)

                result=cv2.warpPerspective(img,matrix,( T[1],T[0]))

                text=pytesseract.image_to_string(result)

                cv2.circle(img,(x1,y1),5,(255,0,0),-1)

                cv2.circle(img,(x2,y1),5,(255,0,0),-1)

                cv2.circle(img,(x1,y2),5,(255,0,0),-1)
                
                cv2.circle(img,(x2,y2),5,(255,0,0),-1)

                cv2.putText(img,str(text),(140,450),cv2.FONT_HERSHEY_PLAIN,6,(0,255,0),4)
                
            
    cv2.imshow('image',img)
    cv2.imshow("Result",result)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release
cv2.destroyAllWindows()