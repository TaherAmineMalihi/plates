import cv2
from predict_results_plates import YOLO_Pred
import easyocr
import os
import numpy as np
import pandas as pd


yolo = YOLO_Pred('C:/Users/tmalihi/Desktop/Plates_IA/Model/weights/best.onnx',"C:/Users/tmalihi/Desktop/Plates_IA/data.yaml")

image = cv2.imread("C:/Users/tmalihi/Desktop/Plates_IA/image2.jpg")


width = 1200
height = 700
object_count = {}


image = cv2.resize(image, (width, height))
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#predictions image
img_pred, object_count, crop, classes = yolo.predictions(image)
cv2.imshow('prediction_image',image)


#objects counted
print('Object count:')
for class_name, count in object_count.items():
    print(f'{class_name}: {count}')

#condition
    if object_count[class_name] > 20:
            print("the road is bad")
    object_count[class_name] = 0


cv2.waitKey(0)
cv2.destroyAllWindows()



crop = cv2.resize(crop,(500,100))
cv2.imshow("crop", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

reader = easyocr.Reader(['en'], gpu = True)
results = reader.readtext(crop)
text = ""
for i in results:
    text = text + i[1] + " "
print(text)