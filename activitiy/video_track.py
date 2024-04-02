<<<<<<< Updated upstream
import random
=======
import os.path
import random
import sys
import threading
import time
>>>>>>> Stashed changes
from collections import defaultdict
import cv2
from ultralytics import YOLO
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification
<<<<<<< Updated upstream
=======
from  face_detection.compare import get_names_list  # Importujemy funkcję z pliku compare
sys.path.append(os.path.abspath(r"C:\Users\miszel\Desktop\WspolnePrzedsiewziecieInzynierskieC3P4-activity-facedetection"))
>>>>>>> Stashed changes

# load yolov8 model
model_yolo = YOLO('../models/best_today.pt')
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification('czynnosci_latest.pt')

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
center_dict = defaultdict(list)
object_id_list = []
color_dict = {}
max_points = 20  # maksymalna liczba punktów śledzenia

# load video
<<<<<<< Updated upstream
video_path = 'x.mp4'
=======
video_path = 'Projekt M.mp4'
>>>>>>> Stashed changes
cap = cv2.VideoCapture(video_path)

ret = True

def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))

<<<<<<< Updated upstream
frame_count = 0
classifications_dict = {}
persisted_classifications = {}
skip_frames = 10

# read frames
while ret:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

        # Analiza tylko co 5 klatek

        results = model_yolo.track(frame, persist=True)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for box, id in zip(boxes, ids):
                class_id = int(box.cls[0])  # klasa boxa

                if class_id == 2:  # person
                    if frame_count % skip_frames == 0:
                        r = box.xyxy[0].astype(int)
                        x_min, y_min, x_max, y_max = r

                        person_region = frame[y_min:y_max, x_min:x_max]
                        results = detection_keypoint(person_region)

                        if results:
                            results_keypoint = detection_keypoint.get_xy_keypoint(results)
                            input_classification = results_keypoint[10:]
                            results_classification = classification_keypoint(input_classification)

                            classifications_dict[id] = results_classification  # Zapamiętaj klasyfikację dla tej osoby

                        # Sprawdź, czy mamy klasyfikację dla tej osoby i wyświetl ją co 10 klatek
                        if id in classifications_dict and frame_count % 2 == 0:
                            classification_text = classifications_dict[id]
                            persisted_classifications[id] = classification_text

        for id, classification_text in persisted_classifications.items():
            if id in classifications_dict:
                try:
                    index = ids.tolist().index(id)
                    r = boxes[index].xyxy[0].astype(int)
                    x_min, y_min = r[0], r[1]
                    (w, h), _ = cv2.getTextSize(
                        classification_text.upper(),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    # Pozycja tekstu w lewym górnym rogu ramki osoby
                    text_x = x_min
                    text_y = y_min - 4
                    # Upewnij się, że tekst mieści się w obrębie ramki osoby
                    if text_y - h < 0:
                        text_y = y_min + h + 4

                    cv2.putText(frame,
                                f'{classification_text.upper()}',
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,0 , 255),
                                thickness=2
                                )
                except ValueError:
                    # Handle the case where id is not found in ids list
                    print(f"ID {id} not found in ids list. Skipping processing for this id.")

        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
=======
# Zmienna do śledzenia liczby klatek
frame_count = 0
dymy = False
# Pobranie listy imion i identyfikatorów twarzy
names_list = get_names_list()

# Funkcja papcer wykonywana przez pierwszy wątek
def papcer():
    global ret, frame_count, names_list, dymy
    classifications_dict = {}
    persisted_classifications = {}
    skip_frames = 30
    while ret:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            results = model_yolo.track(frame, persist=True)
            for result in results:
                boxes = result.boxes.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    class_id = int(box.cls[0])

                    if class_id == 0:
                        r = box.xyxy[0].astype(int)
                        x_min, y_min, x_max, y_max = r
                        head_region1 = frame[y_min:y_max, x_min:x_max]
                        for name, face_id in names_list:
                            if face_id == id:
                                cv2.putText(frame,
                                            #f'Name: {face_id}',
                                             name,
                                            (x_min, y_min-30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255),
                                            thickness=2
                                            )

                        if frame_count % 10 == 0 and frame_count % 20 != 0:
                            #head_region=cv2.cvtColor(head_region1, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"../face_detection/compare/face_{id}.png", head_region1)

                    if class_id == 2:  # person
                        if frame_count % skip_frames == 0:
                            r = box.xyxy[0].astype(int)
                            x_min, y_min, x_max, y_max = r

                            person_region = frame[y_min:y_max, x_min:x_max]
                            results = detection_keypoint(person_region)

                            if results:
                                results_keypoint = detection_keypoint.get_xy_keypoint(results)
                                input_classification = results_keypoint[10:]
                                results_classification = classification_keypoint(input_classification)

                                classifications_dict[id] = results_classification

                            if id in classifications_dict and frame_count % 2 == 0:
                                classification_text = classifications_dict[id]
                                persisted_classifications[id] = classification_text

            for id, classification_text in persisted_classifications.items():
                if id in classifications_dict:
                    try:
                        index = ids.tolist().index(id)
                        r = boxes[index].xyxy[0].astype(int)
                        x_min, y_min = r[0], r[1]
                        (w, h), _ = cv2.getTextSize(
                            classification_text.upper(),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        text_x = x_min
                        text_y = y_min - 4

                        if text_y - h < 0:
                            text_y = y_min + h + 4

                        cv2.putText(frame,
                                    f'{classification_text.upper()}',
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,0 , 255),
                                    thickness=2
                                    )
                    except ValueError:
                        print(f"ID {id} not found in ids list. Skipping processing for this id.")

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if frame_count % 10 == 0:
                t2 = threading.Thread(target=compare)
                t2.start()

    cap.release()
    cv2.destroyAllWindows()

# Funkcja compare wykonywana przez drugi wątek
def compare():
    global names_list
    names_list = get_names_list()
    print(names_list)

# Tworzenie wątku dla funkcji papcer
t1 = threading.Thread(target=papcer)

# Uruchomienie wątku
t1.start()

# Oczekiwanie na zakończenie wątku
t1.join()
>>>>>>> Stashed changes
