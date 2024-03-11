import numpy as np
from PIL import Image, ImageTk

import cv2
from ultralytics import YOLO


def photo_detect(image_path, model_path, selected_areas):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)

    for area in selected_areas:
        if type(area) == list:
            all_points = np.array(area)
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [all_points], (255, 255, 255))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Binarize the mask
            # Extract the region from the frame using the mask
            cropped_region = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            cropped_region = frame[min(area[1], area[3]):max(area[1], area[3]),
                             min(area[0], area[2]):max(area[0], area[2])]

        results = model.predict(source=cropped_region, conf=0.5)

        for result in results:

            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = box.conf.astype(float)
                if class_id == 0:  # Person
                    r = box.xyxy[0].astype(int)
                    x1, y1, x2, y2 = r
                    if type(area[1]) == tuple:
                        x1 += area[0][0] - min(area[0][0], area[1][0])  # Adjust x1
                        x2 += area[1][0] - min(area[0][0], area[1][0])  # Adjust x2
                        y1 += area[0][1] - min(area[0][1], area[1][1])  # Adjust y1
                        y2 += area[1][1] - min(area[0][1], area[1][1])  # Adjust y2
                    else:
                        x1 += min(area[0], area[2])
                        x2 += min(area[0], area[2])
                        y1 += min(area[1], area[3])
                        y2 += min(area[1], area[3])

                    class_name = model.names[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Name: {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                    cv2.putText(frame, f'Confidence: {confidence}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)
                elif class_id == 2:  # Head
                    r = box.xyxy[0].astype(int)
                    x1, y1, x2, y2 = r
                    if type(area[1]) == tuple:
                        x1 += area[0][0] - min(area[0][0], area[1][0])  # Adjust x1
                        x2 += area[1][0] - min(area[0][0], area[1][0])  # Adjust x2
                        y1 += area[0][1] - min(area[0][1], area[1][1])  # Adjust y1
                        y2 += area[1][1] - min(area[0][1], area[1][1])  # Adjust y2
                    else:
                        x1 += min(area[0], area[2])
                        x2 += min(area[0], area[2])
                        y1 += min(area[1], area[3])
                        y2 += min(area[1], area[3])

                    class_name = model.names[class_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, f'Name: {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                    cv2.putText(frame, f'Confidence: {confidence}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

    # Resize the frame for display (optional)


    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.resize(frame_rgb, (800,600))
    # Convert the frame to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Convert PIL Image to Tkinter compatible format
    tk_image = ImageTk.PhotoImage(pil_image)

    return tk_image
