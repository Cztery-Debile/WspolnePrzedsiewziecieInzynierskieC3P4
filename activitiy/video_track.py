import os
import socket
import threading
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from face_detection.compare import  get_names_list
from keras.src.saving import load_model

# load yolov8 model

#================================================Jak jesteś Maciek to daj na true================================================
global WSL
WSL = True
# przechowywanie wykrytych głów i ich obszarów
head_regions_dict = {}
active_head_ids = []

# przechowywanie osob dla czynnosci
classifications_dict = {}

# load video
frame_count = 0

ret = True

# Pobranie listy imion i identyfikatorów twarzy
names_list = get_names_list({})

#Jakbyście to sobie jeszcze chcieli odpalić to odkomentujcie i zakomentujcie tamtą

if WSL:
    def test_predict(frame):
        train_action = pd.read_csv("activitiy/New_model/Training_set.csv")
        # Skalowanie obrazu do wymaganego rozmiaru
        resized_frame = cv2.resize(frame, (224, 224))
        # Konwersja kolorów z BGR na RGB
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', 3500))  # Connect to the WSL script

            # Send the data for calculation
            data = resized_frame_rgb.tobytes()
            s.sendall(data)

            # Receive and print results until the connection is closed
            prediction_result = s.recv(16)
            if not prediction_result:
                #print("Connection closed by the WSL script.")
                return None, None  # Return None if no prediction result is received
            result = np.frombuffer(prediction_result, dtype=np.float32)

        # Przetwarzanie wyników
        prediction = np.argmax(result)
        confidence = np.max(result) * 100

        unique_labels = train_action['label'].unique()
        label_mapping = {label_id: label_name for label_id, label_name in enumerate(unique_labels)}
        predicted_class = label_mapping[prediction]

        return predicted_class, confidence
else:
    cnn_model = load_model('New_model/human_model.h5')
    def test_predict(frame):
        train_action = pd.read_csv("New_model/Training_set.csv")

        # Skalowanie obrazu do wymaganego rozmiaru
        resized_frame = cv2.resize(frame, (224, 224))

        # Konwersja kolorów z BGR na RGB
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Wykonaj predykcję na przeskalowanym obrazie
        result = cnn_model.predict(np.expand_dims(resized_frame_rgb, axis=0))

        # Przetwarzanie wyników
        prediction = np.argmax(result)
        confidence = np.max(result) * 100
        #print("Probability:", confidence, "%")

        unique_labels = train_action['label'].unique()
        label_mapping = {label_id: label_name for label_id, label_name in enumerate(unique_labels)}

        predicted_class = label_mapping[prediction]
        #print(predicted_class)
        return predicted_class, confidence


# Funkcja papcer wykonywana przez pierwszy wątek
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

            results = model_yolo.track(frame, persist=True, device=0, verbose=False)
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

                        # cv2.rectangle(frame, person_box[:2], person_box[2:], (0, 255, 0), 2)

                        # cv2.putText(frame, f'{id}', (person_box[0] + 50, person_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        #             2,
                        #             (255, 0, 0), 3)

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

                                    # wykrywanie czynnosci

                                    if frame_count % 15 == 0:
                                        person_region = frame[person_y_min:person_y_max, person_x_min:person_x_max]
                                        pred = test_predict(person_region)
                                        classifications_dict[id] = pred


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
                            #print(head_regions_dict[id])

            # #jesli w liscie znajduje sie id to znaczy ze wykryto czynnosc dla osoby
            for id, pred in classifications_dict.items():
                try:
                    # Get the coordinates of the person from the detection results
                    box_indices = np.where(ids == id)[0]
                    if len(box_indices) > 0:
                        box_index = box_indices[0]
                        r = boxes[box_index].xyxy[0].astype(int)

                        # wypysanie nazwy wykrytej osoby
                        for face_id, (name, _) in names_list.items():
                            if face_id == id:
                                cv2.putText(frame,
                                            name,
                                            (r[0], r[1] - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255),
                                            thickness=2
                                            )

                        cv2.putText(frame,
                                    f'{pred[0]}',
                                    (r[0] + 20, r[1] + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0),
                                    thickness=2
                                    )

                        cv2.putText(frame,
                                    f'{round(pred[1], 2)}',
                                    (r[0] + 20, r[1] + 130),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255),
                                    thickness=2
                                    )
                    else:
                        continue
                     #   print(f"No box found for ID {id}. Skipping processing for this ID.")
                except ValueError as e:
                    continue
                    # Handle the situation where the ID is not found in the list of ids
                    #print(f"ID {id} was not found in the list of ids. Skipping processing for this ID.")


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
    #print(names_list)

def delete_images():
    folder_path = 'face_detection/compare/'

    files_to_delete = os.listdir(folder_path)

    # Iteruj przez każdy element w liście
    for file in files_to_delete:
        full_path = os.path.join(folder_path, file)
        os.remove(full_path)

#
# # Tworzenie wątku dla funkcji papcer
# t1 = threading.Thread(target=analyze_video)
#
# # Uruchomienie wątku
# t1.start()
#
# # Oczekiwanie na zakończenie wątku
# t1.join()