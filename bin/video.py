import random
from collections import defaultdict
import torch
from ultralytics import YOLO
import cv2

from bin.make_report import make_report


def detect_from_video(model, video_path):
    # load yolov8 model
    model = YOLO(model)

    cap = cv2.VideoCapture(video_path)

    center_dict = defaultdict(list)
    object_id_list = []
    color_dict = {}
    max_points = 50  # maksymalna liczba punktów śledzenia
    all_detect_count = 0


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
            # detect objects
            # track objects

            results = model.track(frame, persist=True)
            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    resized_frame = cv2.resize(frame, (1140, 740))
                    cv2.imshow("act", resized_frame)
                else:
                    boxes = result.boxes.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)

                    class_ids = [int(box.cls[0]) for box in boxes]  # lista id potrzebna do liczenia
                    all_detect_count = len(class_ids)

                    cv2.putText(frame, f'Count: {all_detect_count}', (10,50), cv2.FONT_HERSHEY_TRIPLEX, 2,
                                (0, 255, 255), 3)

                    if len(class_ids) > all_detect_count:
                        all_detect_count = len(class_ids)

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



                        if class_id == 2:  # person
                            r = box.xyxy[0].astype(int)
                            draw_lines(r, id, frame)  # rysowanie lini
                            class_name = model.names[class_id]
                            cv2.rectangle(frame, r[:2], r[2:], color_dict[id], 2)  # rysowanie na obrazie boxa
                            cv2.putText(frame, f'Name: {id}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        2)
                            cv2.imshow(f"Person {id}", frame[r[1]:r[3], r[0]:r[2]]) #to wyswietla boxy


                resized_frame = cv2.resize(frame, (1140, 740))
                cv2.imshow("act", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                make_report(None, video_path, all_detect_count, True)
                break