import csv
import datetime
import glob
import os
import random

import numpy as np
import torch.cuda
from PIL import Image, ImageTk
import cv2
from pyexcel import merge_all_to_a_book
from ultralytics import YOLO

from bin.make_report import make_report


def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))


def photo_detect(image_path, model_path, selected_areas, process_whole_frame=False, width=0, height=0):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    all_results = [] # lista zawierająca wszystkie wyniki detekcji
    fragment_coords_list = [] # lista zawierająca współrzędne danych obszarów
    all_detect_count = 0 # liczba wszystkich wykrytych obiektów
    detected_objects = {}

    frame_resized = cv2.resize(frame, (width, height))
    height_ratio = float(height / frame.shape[0])
    width_ratio = float(width / frame.shape[1])
    if process_whole_frame:
        for y in range(0, frame.shape[0], 160):
            for x in range(0, frame.shape[1], 160):
                fragment_coords_list.append((x, y))
                fragment = frame[y:y + 160, x:x + 160]

                # jesli jest karta graficzna
                if torch.cuda.is_available():
                    results = model.predict(source=fragment, max_det=10000, half=True,
                                            augment=True, iou=0.2, device=0, conf=0.55)
                else:
                    results = model.predict(source=fragment, max_det=10000,
                                            augment=True, iou=0.2)

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    color = random_color()

                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf.astype(float)

                        if class_id == 0:  # Person
                            r = box.xyxy[0].astype(int)
                            x1, y1, x2, y2 = r

                            x1 = int((x1 + x) * width_ratio)
                            x2 = int((x2 + x) * width_ratio)
                            y1 = int((y1 + y) * height_ratio)
                            y2 = int((y2 + y) * height_ratio)

                            # Check if this object has already been detected
                            if (x1, y1, x2, y2) not in detected_objects.values():
                                class_name = model.names[class_id]
                                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                                # cv2.putText(frame, f'Name: {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                # cv2.putText(frame, f'Confidence: {confidence}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                all_detect_count += 1

                                # Add the detected object and its bounding box to the dictionary
                                detected_objects[all_detect_count] = (x1, y1, x2, y2)
    else:
        for i, area in enumerate(selected_areas):
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

            # If cropped region size is larger than 640x640, divide it into smaller fragments
            if cropped_region.shape[0] > 640 or cropped_region.shape[1] > 640:
                fragment_height = min(cropped_region.shape[0], 320)
                fragment_width = min(cropped_region.shape[1], 320)
                for y in range(0, cropped_region.shape[0], fragment_height):
                    for x in range(0, cropped_region.shape[1], fragment_width):
                        # Store coordinates of the fragment
                        fragment_coords_list.append((x, y))
                        fragment = cropped_region[y:y + fragment_height, x:x + fragment_width]
                        if torch.cuda.is_available():
                            results = model.predict(source=fragment, max_det=10000, half=True,
                                                    augment=True, iou=0.2, device=0, conf=0.55)
                        else:
                            results = model.predict(source=fragment, conf=0.55, max_det=10000,
                                                    augment=True, iou=0.2)

                        all_results.extend([(i, fragment.shape[:2], result, (x, y), True) for result in results])
            else:
                if torch.cuda.is_available():
                    results = model.predict(source=cropped_region, max_det=10000, half=True,
                                            augment=True, iou=0.2, device=0)
                else:
                    results = model.predict(source=cropped_region, max_det=10000,
                                            augment=True, iou=0.2)

                fragment_coords_list.append((0, 0))
                all_results.extend([(i, cropped_region.shape[:2], result, (0, 0), False) for result in results])

            for result_info in all_results:
                area_index, area_size, result, fragment_coords, check_for_fragments = result_info
                boxes = result.boxes.cpu().numpy()
                color = random_color()
                class_ids = [int(box.cls[0]) for box in boxes] # lista id potrzebna do liczenia
                all_detect_count += len(class_ids)

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf.astype(float)

                    if class_id == 0:  # Person
                        r = box.xyxy[0].astype(int)
                        x1, y1, x2, y2 = r

                        x1 = int(x1 * width_ratio)
                        x2 = int(x2 * width_ratio)
                        y1 = int(y1 * height_ratio)
                        y2 = int(y2 * height_ratio)

                        if result_info[4]:
                            if type(selected_areas[area_index][1]) == tuple:
                                print("1")
                                x1 = x1 + int(fragment_coords[0] * width_ratio) + (
                                    int(selected_areas[area_index][0][0] * width_ratio)
                                    - min(int(selected_areas[area_index][0][0] * width_ratio),
                                          int(selected_areas[area_index][1][0] * width_ratio)))
                                x2 = x2 + int(fragment_coords[0] * width_ratio) + (
                                    int(selected_areas[area_index][1][0] * width_ratio)
                                    - min(int(selected_areas[area_index][0][0] * width_ratio),
                                          int(selected_areas[area_index][1][0] * width_ratio)))
                                y1 = y1 + int(fragment_coords[1] * height_ratio) + (
                                    int(selected_areas[area_index][0][1] * height_ratio)
                                    - min(int(selected_areas[area_index][0][1] * height_ratio),
                                          int(selected_areas[area_index][1][1] * height_ratio)))
                                y2 = y2 + int(fragment_coords[1] * height_ratio) + (
                                    int(selected_areas[area_index][1][1] * height_ratio)
                                    - min(int(selected_areas[area_index][0][1] * height_ratio),
                                          int(selected_areas[area_index][1][1] * height_ratio)))
                            else:
                                print("2")
                                x1 = x1 + int(fragment_coords[0] * width_ratio) + min(
                                    int(selected_areas[area_index][0] * width_ratio),
                                    int(selected_areas[area_index][2] * width_ratio))
                                x2 = x2 + int(fragment_coords[0] * width_ratio) + min(
                                    int(selected_areas[area_index][0] * width_ratio),
                                    int(selected_areas[area_index][2] * width_ratio))
                                y1 = y1 + int(fragment_coords[1] * height_ratio) + min(
                                    int(selected_areas[area_index][1] * height_ratio),
                                    int(selected_areas[area_index][3] * height_ratio))
                                y2 = y2 + int(fragment_coords[1] * height_ratio) + min(
                                    int(selected_areas[area_index][1] * height_ratio),
                                    int(selected_areas[area_index][3] * height_ratio))
                        else:
                            if type(selected_areas[area_index][1]) == tuple:
                                print("3")
                                x1 += int(selected_areas[area_index][0][0] * width_ratio) - min(
                                    int(selected_areas[area_index][0][0] * width_ratio),
                                    int(selected_areas[area_index][1][0] * width_ratio))
                                x2 += int(selected_areas[area_index][1][0] * width_ratio) - min(
                                    int(selected_areas[area_index][0][0] * width_ratio),
                                    int(selected_areas[area_index][1][0] * width_ratio))
                                y1 += int(selected_areas[area_index][0][1] * height_ratio) - min(
                                    int(selected_areas[area_index][0][1] * height_ratio),
                                    int(selected_areas[area_index][1][1] * height_ratio))
                                y2 += int(selected_areas[area_index][1][1] * height_ratio) - min(
                                    int(selected_areas[area_index][0][1] * height_ratio),
                                    int(selected_areas[area_index][1][1] * height_ratio))
                            else:
                                print("4")
                                x1 += min(int(selected_areas[area_index][0] * width_ratio),
                                          int(selected_areas[area_index][2] * width_ratio))
                                x2 += min(int(selected_areas[area_index][0] * width_ratio),
                                          int(selected_areas[area_index][2] * width_ratio))
                                y1 += min(int(selected_areas[area_index][1] * height_ratio),
                                          int(selected_areas[area_index][3] * height_ratio))
                                y2 += min(int(selected_areas[area_index][1] * height_ratio),
                                          int(selected_areas[area_index][3] * height_ratio))

                        # Apply the resizing factors
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)

                        class_name = model.names[class_id]
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                        # cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                        # cv2.putText(frame, f'Name: {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        #             2)
                        # cv2.putText(frame, f'Confidence: {confidence}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (255, 0, 0), 2)


    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # if height > 0 and width > 0:
    #     pil_image = pil_image.resize((width,height))

    # Convert PIL Image to Tkinter compatible format
    tk_image = ImageTk.PhotoImage(pil_image)

    print(all_detect_count)

    # tworzenie raportu
    make_report(None, image_path, all_detect_count, False)

    return tk_image
