import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
detector =HandDetector(maxHands=1) # using 1 hand

cap =cv2.VideoCapture(0)

offset =20
img_Size=300


folder ="Data\Rock n role"
counter =0
while True:
    success,img =cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand =hands[0]
        x, y, width, height =hand['bbox'] # height weight bounding box
        img_White = np.ones((img_Size, img_Size,3),np.uint8)*255 # white background, square image,data types 8 bit value
        img_Crop = img[y- offset :y+height+offset, x:x+width+offset]
        image_CropShape =img_Crop.shape

# if width is larger than 300, reduce it below 300. and for height vice versa

        aspectRatio = height/width
        # for height
        if aspectRatio >1:
            R = img_Size/height
            w_Cal =math.ceil(R*width)
            img_Resize =cv2.resize(img_Crop,(w_Cal,img_Size))
            img_Resize_Shape =img_Resize.shape
            w_Gap = math.ceil((img_Size-w_Cal)/2) # centre the image
            img_White[:, w_Gap:w_Cal+w_Gap] = img_Resize
            # for witdth
        else:
            k = img_Size / width
            h_Cal = math.ceil(R * height)
            img_Resize = cv2.resize(img_Crop, (img_Size,h_Cal))
            img_Resize_Shape = img_Resize.shape
            h_Gap = math.ceil((img_Size - h_Cal) / 2)
            img_White[h_Gap:h_Cal + h_Gap,: ] = img_Resize

        cv2.imshow("ImageCrop", img_Crop)
        cv2.imshow("ImageWhite", img_White)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("a"): # pressing a will save picture
        counter +=1 # counting the image
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)