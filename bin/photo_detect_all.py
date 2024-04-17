import random
import numpy as np
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))

def photo_detect(image_path, model_path):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    all_detect_count = 0

    results = model.predict(source=frame, conf=0.7, max_det=10000)
    all_results = [(0, frame.shape[:2], result, (0, 0), False) for result in results]

    for result_info in all_results:
        area_index, area_size, result, fragment_coords, check_for_fragments = result_info
        boxes = result.boxes.cpu().numpy()
        color = random_color()

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = box.conf.astype(float)

            if class_id == 0 or class_id==2:  # Person
                r = box.xyxy[0].astype(int)
                x1, y1, x2, y2 = r

                class_name = model.names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, f'Name: {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f'Confidence: {confidence}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                all_detect_count += 1

    # Resize the frame for display (optional)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Convert PIL Image to Tkinter compatible format
    tk_image = ImageTk.PhotoImage(pil_image)

    print(all_detect_count)

    return tk_image