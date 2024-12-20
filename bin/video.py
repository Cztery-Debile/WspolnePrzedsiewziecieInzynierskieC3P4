import random
from collections import defaultdict
from datetime import datetime, timedelta

import torch
import keyboard
from ultralytics import YOLO
import cv2
from ultralytics.solutions import heatmap
from bin.make_report import make_report

heatmap_generation = False
def detect_from_video(video_path,model, centered):
    # load yolov8 model
    model = YOLO(model)
    cap = cv2.VideoCapture(video_path)

    center_dict = defaultdict(list)
    object_id_list = []
    color_dict = {}
    max_points = 50  # maksymalna liczba punktów śledzenia

    #count people
    all_detect_count = 0
    max_count_interval = 0
    interval_start_time = datetime.now()  # initial sys time
    interval_counts = []  # list to store time : max detect count

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
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # read frames
    while ret:
        ret, frame = cap.read()
        if ret:
            # detect objects
            # track objects
            if centered == "true":
                imgsz = (1920,1088)
            elif centered == "false":
                imgsz = (1088,1920)

            if torch.cuda.is_available():
                results = model.track(frame, persist=True, imgsz=imgsz,
                                      augment=True, iou=0.1, max_det=10000, device=0, half=True)
            else:
                results = model.track(frame, persist=True)

            for result in results:
                current_time = datetime.now()

                if (current_time - interval_start_time) >= timedelta(minutes=30):
                    interval_counts.append((interval_start_time.strftime('%H:%M'), max_count_interval))

                    max_count_interval = 0  # reset interval count
                    interval_start_time = current_time  # set current time

                if result.boxes is None or result.boxes.id is None:
                    resized_frame = cv2.resize(frame, (1140, 740))
                    cv2.imshow("act", resized_frame)
                else:
                    boxes = result.boxes.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)

                    class_ids = [int(box.cls[0]) for box in boxes]  # lista id potrzebna do liczenia
                    all_detect_count_frame = len(class_ids)

                    cv2.putText(frame, f'Wykryto: {all_detect_count_frame}', (10,50), cv2.FONT_HERSHEY_TRIPLEX, 2,
                                (0, 0, 0), 10)
                    cv2.putText(frame, f'Wykryto: {all_detect_count_frame}', (10,50), cv2.FONT_HERSHEY_TRIPLEX, 2,
                                (255, 255, 255), 3)

                    if len(class_ids) > all_detect_count:
                        all_detect_count = len(class_ids)

                    if len(class_ids) > max_count_interval:
                        max_count_interval = len(class_ids)

                    for box, id in zip(boxes, ids):
                        class_id = int(box.cls[0])  # klasa boxa

                        if class_id == 0:  # tlumy person
                            r = box.xyxy[0].astype(int)  #
                            class_name = model.names[class_id]  #
                            confidence = box[0].conf.astype(float)

                            cv2.rectangle(frame, r[:2], r[2:], (0, 255, 0), 2)  # rysowanie na obrazie boxa
                            # cv2.putText(frame, f'{id}', (r[0] + 50, r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            #             (255, 0, 0), 3)
                            # cv2.putText(frame, f'{confidence}', (r[0] + 10, r[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            #             (255, 0, 0), 2)



                        if class_id == 3:  # person
                            r = box.xyxy[0].astype(int)
                            draw_lines(r, id, frame)  # rysowanie lini
                            class_name = model.names[class_id]
                            cv2.rectangle(frame, r[:2], r[2:], color_dict[id], 2)  # rysowanie na obrazie boxa
                            cv2.putText(frame, f'Name: {id}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        2)
                            #cv2.imshow(f"Person {id}", frame[r[1]:r[3], r[0]:r[2]]) #to wyswietla boxy

                resized_frame = cv2.resize(frame, (1140, 740))

                cv2.imshow("act", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                make_report(None, video_path, interval_counts, True)
                break