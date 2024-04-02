import os.path
import random
import sys
import threading
import time
from collections import defaultdict
import cv2
from ultralytics import YOLO
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification
sys.path.append(os.path.abspath(r"C:\Users\kszcz\OneDrive\Pulpit\przedsięwzięcie\WspolnePrzedsiewziecieInzynierskieC3P4\face_detection"))
from face_detection.compare import get_names_list  # Importujemy funkcję z pliku compare

# load yolov8 model
model_yolo = YOLO('../models/best_today.pt')
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification('czynnosci_latest.pt')

center_dict = defaultdict(list)
object_id_list = []
color_dict = {}

# load video
video_path = 'Projekt M - Trim.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))

# Zmienna do śledzenia liczby klatek
frame_count = 0

# Pobranie listy imion i identyfikatorów twarzy
names_list = get_names_list()

# Funkcja papcer wykonywana przez pierwszy wątek
def papcer():
    global ret, frame_count, names_list
    classifications_dict = {}
    persisted_classifications = {}
    skip_frames = 10
    while ret:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            results = model_yolo.track(frame, persist=True, device=0)
            for result in results:
                boxes = result.boxes.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    class_id = int(box.cls[0])

                    if class_id == 0:
                        r = box.xyxy[0].astype(int)
                        x_min, y_min, x_max, y_max = r
                        head_region = frame[y_min:y_max, x_min:x_max]
                        for name, face_id in names_list:
                            if face_id == id:
                                cv2.putText(frame,
                                            name,
                                            (x_min, y_min-30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255),
                                            thickness=2
                                            )

                        if frame_count % 35 == 0 :
                            cv2.imwrite(f"../face_detection/compare/face_{id}.png", head_region)



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

            if frame_count % 35 == 0 :
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