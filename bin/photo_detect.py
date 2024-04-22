import csv
import datetime
import glob
import os
import random

import numpy as np
from PIL import Image, ImageTk
import cv2
from pyexcel import merge_all_to_a_book
from ultralytics import YOLO


def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))

def generate_report(all_detect_count, output_file, image_path):
    with open(output_file, 'w', newline='', encoding='UTF8') as csvfile:
        fieldnames = ['Nazwa Pliku', 'Łączna ilość wykrytych osób']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Nazwa Pliku' : image_path ,  'Łączna ilość wykrytych osób': all_detect_count})

    merge_all_to_a_book(glob.glob(output_file), f"{output_file.rstrip('.csv')}.xlsx")
    os.startfile(output_file.rstrip(".csv")+".xlsx")

def photo_detect(image_path, model_path, selected_areas, process_whole_frame=False):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    all_results = [] # lista zawierająca wszystkie wyniki detekcji
    fragment_coords_list = [] # lista zawierająca współrzędne danych obszarów
    all_detect_count = 0 # liczba wszystkich wykrytych obiektów
    detected_objects = {}
    if process_whole_frame:
        for y in range(0, frame.shape[0], 160):
            for x in range(0, frame.shape[1], 160):
                fragment_coords_list.append((x, y))
                fragment = frame[y:y + 160, x:x + 160]

                results = model.predict(source=fragment, conf=0.55, max_det=10000)

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    color = random_color()

                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf.astype(float)

                        if class_id == 0:  # Person
                            r = box.xyxy[0].astype(int)
                            x1, y1, x2, y2 = r

                            x1 += x  # Adjust x1
                            x2 += x  # Adjust x2
                            y1 += y  # Adjust y1
                            y2 += y  # Adjust y2

                            # Check if this object has already been detected
                            if (x1, y1, x2, y2) not in detected_objects.values():
                                class_name = model.names[class_id]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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
                        results = model.predict(source=fragment, conf=0.55, max_det=10000)
                        all_results.extend([(i, fragment.shape[:2], result, (x, y), True) for result in results])
            else:
                results = model.predict(source=cropped_region, conf=0.7, max_det=10000)
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
                        if result_info[4]:
                            if type(selected_areas[i][1]) == tuple:
                                x1 = x1 + fragment_coords[0] + (selected_areas[i][0][0] - min(selected_areas[i][0][0], selected_areas[i][1][0]))
                                x2 = x2 + fragment_coords[0] + (selected_areas[i][1][0] - min(selected_areas[i][0][0], selected_areas[i][1][0]))
                                y1 = y1 + fragment_coords[1] + (selected_areas[i][0][1] - min(selected_areas[i][0][1], selected_areas[i][1][1]))
                                y2 = y2 + fragment_coords[1] + (selected_areas[i][1][1] - min(selected_areas[i][0][1], selected_areas[i][1][1]))
                            else:
                                x1 = x1 + fragment_coords[0] + min(selected_areas[i][0], selected_areas[i][2])
                                x2 = x2 + fragment_coords[0] + min(selected_areas[i][0], selected_areas[i][2])
                                y1 = y1 + fragment_coords[1] + min(selected_areas[i][1], selected_areas[i][3])
                                y2 = y2 + fragment_coords[1] + min(selected_areas[i][1], selected_areas[i][3])
                        else:
                            if type(selected_areas[i][1]) == tuple:
                                x1 += selected_areas[i][0][0] - min(selected_areas[i][0][0], selected_areas[i][1][0])  # Adjust x1
                                x2 += selected_areas[i][1][0] - min(selected_areas[i][0][0], selected_areas[i][1][0])  # Adjust x2
                                y1 += selected_areas[i][0][1] - min(selected_areas[i][0][1], selected_areas[i][1][1])  # Adjust y1
                                y2 += selected_areas[i][1][1] - min(selected_areas[i][0][1], selected_areas[i][1][1])  # Adjust y2
                            else:
                                x1 += min(selected_areas[i][0], selected_areas[i][2])
                                x2 += min(selected_areas[i][0], selected_areas[i][2])
                                y1 += min(selected_areas[i][1], selected_areas[i][3])
                                y2 += min(selected_areas[i][1], selected_areas[i][3])

                        class_name = model.names[class_id]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # cv2.putText(frame, f'Name: {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        #             2)
                        # cv2.putText(frame, f'Confidence: {confidence}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (255, 0, 0), 2)

    # Resize the frame for display (optional)


    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Convert PIL Image to Tkinter compatible format
    tk_image = ImageTk.PhotoImage(pil_image)

    print(all_detect_count)

    # tworzenie raportu
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")
    folder_path = os.path.join('reports/photos', formatted_time)
    os.makedirs(folder_path, exist_ok=True)
    output_file = os.path.join(folder_path, f'p_report_{formatted_time}.csv')
    generate_report(all_detect_count, output_file, image_path)

    print(image_path)

    return tk_image