import datetime
import os
import random
import numpy as np
import torch
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

from bin.make_report import make_report


def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))

def photo_detect_all(image_path, model_path, width=0, height=0):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    frame_resized = cv2.resize(frame, (width,height))
    sizes = (frame.shape[0], frame.shape[1])
    all_detect_count = 0
    height_ratio=float(height/frame.shape[0])
    width_ratio=float(width/frame.shape[1])
    if torch.cuda.is_available():
        results = model.predict(source=frame, max_det=10000, half=True,
                                augment=True, iou=0.2, device=0, imgsz=sizes)
    else:
        results = model.predict(source=frame, max_det=10000,
                                augment=True, iou=0.2, imgsz=sizes)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        color = random_color()

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = box.conf.astype(float)

            if class_id == 0:  # Person
                r = box.xyxy[0].astype(int)
                x1, y1, x2, y2 = r
                cv2.rectangle(frame_resized, (int(x1*width_ratio), int(y1*height_ratio)), (int(x2*width_ratio),
                                                                                           int(y2*height_ratio)), color, 2)
                all_detect_count += 1

    # Resize the frame for display (optional)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # if height > 0 and width > 0:
    #     pil_image = pil_image.resize((width,height))
    # Convert PIL Image to Tkinter compatible format
    tk_image = ImageTk.PhotoImage(pil_image)
    #frame_resized=cv2.resize(frame,(1920,1080))
    # tworzenie raportu
    make_report(None, image_path, all_detect_count, False)

    return tk_image