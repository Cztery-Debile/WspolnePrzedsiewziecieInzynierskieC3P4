import csv
import glob
import os
import socket
import threading
import cv2
import numpy as np
import pandas as pd
from pyexcel import merge_all_to_a_book
from ultralytics import YOLO
from face_detection.compare import  get_names_list

# przechowywanie wykrytych głów i ich obszarów
head_regions_dict = {}
active_head_ids = []

# load video
frame_count = 0

ret = True

# Pobranie listy imion i identyfikatorów twarzy
names_list = get_names_list({})


def analyze_video(model_path,video_path):
    compare()
    model_yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    global ret, frame_count, names_list
    head_assigned = {}
    while ret:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            results = model_yolo.track(frame, persist=True, verbose=True)
            for result in results:
                boxes = result.boxes.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    class_id = int(box.cls[0])

                    if class_id == 0:
                        r = box.xyxy[0].astype(int)
                        x_min, y_min, x_max, y_max = r

                        #dodanie obszaru glowy do slownika
                        head_regions_dict[id] = (x_min, y_min, x_max, y_max, id)

                        if id not in active_head_ids:
                            active_head_ids.append(id)

                        # cv2.rectangle(frame, r[:2], r[2:], (0, 255, 0), 2)

                    if class_id == 2:  # Person
                        person_box = box.xyxy[0].astype(int)
                        person_x_min, person_y_min, person_x_max, person_y_max = person_box

                        cv2.rectangle(frame, person_box[:2], person_box[2:], (0, 255, 0), 2)

                        cv2.putText(frame, f'{id}', (person_box[0] + 50, person_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (255, 0, 0), 3)

                        # Find the head region with the highest position within the person's bounding box
                        highest_head_y = float('inf')  # Initialize with infinity for comparison
                        highest_head_region = None

                        for stored_id in active_head_ids:
                            if stored_id in head_regions_dict:
                                head_region_coords = head_regions_dict[stored_id]
                                x_min, y_min, x_max, y_max, person_id = head_region_coords

                                head_region = frame[y_min:y_max, x_min:x_max]

                                # sprawdzenie czy głowa zawiera sie wewnatrz boxa osoby
                                if (x_min >= person_x_min and y_min >= person_y_min and
                                        x_max <= person_x_max and y_max <= person_y_max):

                                    # sprawdzenie czy glowa jest najwyzej w boxie osoby
                                    if y_min < highest_head_y:
                                        highest_head_y = y_min
                                        highest_head_region = head_region_coords

                                        if frame_count % 70 == 0:
                                            cv2.imwrite(f"face_detection/compare/face_{id}.png", head_region)

                        if highest_head_region is not None:
                            # Update head region only if it's the latest position for the person's bounding box
                            if id not in head_assigned or head_assigned[id] != highest_head_region[-1]:
                                head_regions_dict[id] = (*highest_head_region[:-1], id)
                                head_assigned[id] = highest_head_region[-1]

                        for face_id, (name, _) in names_list.items():
                            if face_id == id:
                                cv2.putText(frame,
                                            name,
                                            (person_box[0], person_box[1] - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255),
                                            thickness=2
                                            )

            cv2.imshow("Frame", frame)

            # odpalanie funkcji compare
            if frame_count % 70 == 0:
                t2 = threading.Thread(target=compare)
                t2.start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                delete_images()
                break

    cap.release()
    cv2.destroyAllWindows()

# Funkcja compare wykonywana przez drugi wątek
def compare():
    global names_list
    names_list = get_names_list(names_list)

def delete_images():
    folder_path = 'face_detection/compare/'

    files_to_delete = os.listdir(folder_path)

    # Iteruj przez każdy element w liście
    for file in files_to_delete:
        full_path = os.path.join(folder_path, file)
        os.remove(full_path)
