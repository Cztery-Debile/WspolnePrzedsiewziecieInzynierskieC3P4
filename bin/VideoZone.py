import csv
import datetime
import os
import random
import threading
from collections import defaultdict
import cv2
import numpy as np
from pyexcel.cookbook import merge_all_to_a_book
import glob
from ultralytics import YOLO

from face_detection.compare import get_names_list

# Pobranie listy imion i identyfikatorów twarzy
names_list = get_names_list({})


def detect_from_video_zone(video_path, model):
    compare()
    model_yolo = YOLO(model)

    cap = cv2.VideoCapture(video_path)
    people_in_zone = {}

    # zmienne do rysowania lini zmiany pozycji
    center_dict = defaultdict(list)
    color_dict = {}
    object_id_list = []
    max_points = 50

    # zmienne do przechowywania obszarów wykrytych głow
    head_regions_dict = {}
    active_head_ids = []
    head_assigned = {}

    # obszar wykrywania
    specified_region = (700, 280, 780, 350)

    person_time = {}
    frame_count = 0

    def random_color():
        return tuple(int(random.random() * 255) for _ in range(3))

    def draw_lines(r, id, frame):
        x1 = int(r[0])
        y1 = int(r[1])
        x2 = int(r[2])
        y2 = int(r[3])

        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)

        if id not in color_dict:
            color_dict[id] = random_color()  # mapowanie koloru dla danego ID wykrytej osoby

        cv2.circle(frame, (cX, cY), 4, color_dict[id], -1)

        center_dict[id].append((cX, cY))  # mapowanie srodka dla danego ID wykrytej osoby
        if id not in object_id_list:
            object_id_list.append(id)
            start_pt = (cX, cY)
            end_pt = (cX, cY)
            cv2.line(frame, start_pt, end_pt, color_dict[id], 7)
        else:
            l = len(center_dict[id])
            if l > max_points:  # Sprawdzenie, czy przekroczono maksymalną liczbę punktów
                center_dict[id] = center_dict[id][1:]  # Usunięcie najstarszego punktu
            for pt in range(len(center_dict[id])):
                if len(center_dict[id]) > pt + 1:  # Sprawdzenie, czy indeks istnieje w liście
                    start_pt = (center_dict[id][pt][0], center_dict[id][pt][1])
                    end_pt = (center_dict[id][pt + 1][0], center_dict[id][pt + 1][1])
                    cv2.line(frame, start_pt, end_pt, color_dict[id], 7)

    ret = True
    # read frames
    while ret:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            cv2.rectangle(frame, (specified_region[0], specified_region[1]), (specified_region[2], specified_region[3]), (0, 255, 0), 2)

            results = model_yolo.track(frame, persist=True, verbose=False)

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    resized_frame = cv2.resize(frame, (1140, 740))
                    cv2.imshow("act", resized_frame)
                else:
                    boxes = result.boxes.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)

                    for box, id in zip(boxes, ids):
                        class_id = int(box.cls[0])  # klasa boxa

                        # jesli jest to glowa zapisz jej obszar do slownika
                        if class_id == 0:
                            r = box.xyxy[0].astype(int)
                            x_min, y_min, x_max, y_max = r

                            # dodanie obszaru glowy do slownika
                            head_regions_dict[id] = (x_min, y_min, x_max, y_max, id)

                            if id not in active_head_ids:
                                active_head_ids.append(id)


                        if class_id == 2:  # person
                            person_box = box.xyxy[0].astype(int)
                            person_x_min, person_y_min, person_x_max, person_y_max = person_box

                            draw_lines(person_box, id, frame)

                            # =============================Przypisywanie najwyzszej glowy do obszaru boxa osoby================

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

                                        if y_min < highest_head_y:
                                            highest_head_y = y_min
                                            highest_head_region = head_region_coords

                                            if frame_count % 60 == 0:
                                                cv2.imwrite(f"face_detection/compare/face_{id}.png", head_region)

                            if highest_head_region is not None:
                                # Update head region only if it's the latest position for the person's bounding box
                                if id not in head_assigned or head_assigned[id] != highest_head_region[-1]:
                                    head_regions_dict[id] = (*highest_head_region[:-1], id)
                                    head_assigned[id] = highest_head_region[-1]


                            #==============================Analiza przebywania w obszarze zona==================================

                            if ((person_x_max >= specified_region[0] and person_y_max >= specified_region[1] and
                                 person_x_min <= specified_region[2] and person_y_min <= specified_region[3])):

                                # sprawdzenie czy id wykrytej osoby pokrywa sie z id twarzy, przypisanie tej osoby do listy osob w zonie wraz z jej imieniem i start czasu
                                for face_id, (name, _) in names_list.items():
                                    if face_id == id:
                                        if id not in people_in_zone:
                                            people_in_zone[id] = name, current_time_ms

                                # jesli osoba znajduje sie w zonie i ma przypisane imie wyswietlaj je (pierwsze imie jakie wykryto w zonie)
                                if id in people_in_zone and people_in_zone[id][0] is not None:
                                    cv2.putText(frame,
                                                people_in_zone[id][0],
                                                (person_box[0], person_box[1] - 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, (0, 0, 255),
                                                thickness=2
                                                )

                            else:
                                # jesli osoba byla w zonie i go opuściała oblicz czas i usun z listy
                                if id in people_in_zone:
                                    time_spent_ms = current_time_ms - people_in_zone[id][1]  # Obliczenie czasu spędzonego
                                    #print(f"{people_in_zone[id][0]}, {id} spędziła {time_spent_ms / 1000} sekund w obszarze.")

                                    if id in person_time:
                                        if time_spent_ms / 1000 > 3:
                                            # nie zmieniaj pierwszego imienia przypisanego do id, dodaj czas, dodaj ilość wykonania jesli więcej niz 3 sekundy
                                            person_time[id] = person_time[id][0], round(person_time[id][1] + (time_spent_ms / 1000), 2), person_time[id][2] + 1
                                    else:
                                        person_time[id] = people_in_zone[id][0], round((time_spent_ms / 1000), 2), 0

                                    del people_in_zone[id]  # Usunięcie osoby z obszaru

                                else:
                                    # wyświetlanie aktulanie przypisanego imienia dla osob poza zonem
                                    for face_id, (name, _) in names_list.items():
                                        if face_id == id:
                                            cv2.putText(frame,
                                                        name,
                                                        (person_box[0], person_box[1] - 30),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.5, (0, 0, 255),
                                                        thickness=2
                                                        )

                # Wyświetlenie dla osoby w zonie
            for id in people_in_zone:
                try:
                    box_indices = np.where(ids == id)[0]
                    if len(box_indices) > 0:
                        box_index = box_indices[0]
                        r = boxes[box_index].xyxy[0].astype(int)

                        # cv2.rectangle(frame, r[:2], r[2:], (255,0,0), 2)  # rysowanie na obrazie boxa
                        # # Display the activity above the person's head
                        # cv2.putText(frame,
                        #             f'{id}',
                        #             (r[0] + 20, r[1] + 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX,
                        #             1, (0, 255, 0),
                        #             thickness=3
                        #             )

                    else:
                        print(f"No box found for ID {id}. Skipping processing for this ID.")
                except ValueError as e:
                    # Handle the situation where the ID is not found in the list of ids
                    print(f"ID {id} was not found in the list of ids. Skipping processing for this ID.")

            cv2.imshow('Frame', frame)

            if frame_count % 60 == 0:
                t2 = threading.Thread(target=compare)
                t2.start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                delete_images()

                # tworzenie raportu
                folder_path = 'reports/videos'
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")
                folder_path = os.path.join('reports/videos', formatted_time)
                os.makedirs(folder_path, exist_ok=True)
                output_file = os.path.join(folder_path, f'p_report_{formatted_time}.csv')
                generate_report(person_time, output_file)
                break


def compare():
    global names_list
    print('stara list:', names_list)
    names_list = get_names_list(names_list)
    print('nowa lista:' , names_list)


def delete_images():
    folder_path = 'face_detection/compare/'

    files_to_delete = os.listdir(folder_path)

    # Iteruj przez każdy element w liście
    for file in files_to_delete:
        full_path = os.path.join(folder_path, file)
        os.remove(full_path) 


def generate_report(person_time, output_file):
    with open(output_file, 'w', newline='', encoding='UTF8') as csvfile:
        fieldnames = ['ID', 'Osoba', 'Całkowity czas (s)', 'Ilość wykonanych rzeczy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for person_id, (person, total_time, num_tasks) in person_time.items():
            writer.writerow({'ID': person_id, 'Osoba': person, 'Całkowity czas (s)': total_time, 'Ilość wykonanych rzeczy': num_tasks})

    merge_all_to_a_book(glob.glob(output_file), f"{output_file.rstrip('.csv')}.xlsx")
    os.startfile(output_file.rstrip(".csv")+".xlsx")

