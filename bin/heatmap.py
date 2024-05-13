from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("../models/tokioKrakau5000.pt")
cap = cv2.VideoCapture('./krakau2.mkv')
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
if not cap.isOpened():
    print("Error opening video stream or file")
    exit(0)


# Init heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                     imw=w,
                     imh=h,
                     view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        exit(0)
    results = model.track(im0, persist=True, device=0)
    frame = heatmap_obj.generate_heatmap(im0, tracks=results)
