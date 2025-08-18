import cv2
import time
from ultralytics import YOLO

model = YOLO("benchmarks/yolo11n.pt")

im2 = cv2.imread("/home/ibra/Documents/iatrax/YOLO-One/datasets/images/test/FudanPed00042_png.rf.868f8c248875a783a8c7033113e2189f.jpg")
start = time.time()
results = model.predict(source=im2, save=False, conf=0.4, imgsz=640, verbose=True, half=True, device="cuda") 
end = time.time()
print("Time taken:{:.2f} seconds".format(end - start))
print("fps: ", 1 / (end - start))
