import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

import time

detector = HandDetector(maxHands=1)

cap = cv2.VideoCapture(0)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
img_Size = 300
#folder = "Data\B"
counter = 0

labels = ["A", "B", "C","D","E","ok","Rock n role"]
while True:
    success, img = cap.read()
    img_Output = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']
        img_White = np.ones((img_Size, img_Size, 3), np.uint8) * 255
        img_Crop = img[y - offset:y + height + offset, x:x + width + offset]
        image_Crop_Shape = img_Crop.shape

        aspectRatio = height / width
        if aspectRatio > 1:
            R = img_Size / height
            w_Cal = math.ceil(R * width)
            img_Resize = cv2.resize(img_Crop, (w_Cal, img_Size))
            img_Resize_Shape = img_Resize.shape
            w_Gap = math.ceil((img_Size - w_Cal) / 2)
            img_White[:, w_Gap:w_Cal + w_Gap] = img_Resize
            prediction, index = classifier.getPrediction(img_White, draw=False)
            print(prediction, index)

        else:
            k = img_Size / width
            h_Cal = math.ceil(R * height)
            img_Resize = cv2.resize(img_Crop, (img_Size, h_Cal))
            img_Resize_Shape = img_Resize.shape
            h_Gap = math.ceil((img_Size - h_Cal) / 2)
            img_White[h_Gap:h_Cal + h_Gap, :] = img_Resize
            prediction, index = classifier.getPrediction(img_White, draw=False)
        cv2.rectangle(img_Output, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255),
                      cv2.FILLED)
        cv2.putText(img_Output, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(img_Output, (x - offset, y - offset), (x + width + offset, y + height + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", img_Crop)
        cv2.imshow("ImageWhite", img_White)
    cv2.imshow("Image", img_Output)
    key = cv2.waitKey(1)
