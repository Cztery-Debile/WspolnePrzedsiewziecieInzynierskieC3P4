import random
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from keras.src.saving import load_model
from ultralytics import YOLO

# load yolov8 model
model_yolo = YOLO('../../models/best_today.pt')
cnn_model = load_model('human_model_last.h5')

# load video
video_path = '../x.mp4'
cap = cv2.VideoCapture(video_path)

ret = True


def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))


frame_count = 0
classifications_dict = {}
skip_frames = 5


def test_predict(frame):
    train_action = pd.read_csv("Training_set.csv")

    # Skalowanie obrazu do wymaganego rozmiaru
    resized_frame = cv2.resize(frame, (224, 224))

    # Konwersja kolorów z BGR na RGB
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Wykonaj predykcję na przeskalowanym obrazie
    result = cnn_model.predict(np.expand_dims(resized_frame_rgb, axis=0))

    # Przetwarzanie wyników
    prediction = np.argmax(result)
    confidence = np.max(result) * 100
    print("Probability:", confidence, "%")

    unique_labels = train_action['label'].unique()
    label_mapping = {label_id: label_name for label_id, label_name in enumerate(unique_labels)}

    predicted_class = label_mapping[prediction]
    print(predicted_class)
    return predicted_class , confidence


# read frames
while ret:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

        frame_resized = cv2.resize(frame, (1280, 720))
        # Analiza tylko co 5 klatek

        if frame_count % skip_frames == 0:
            results = model_yolo.track(frame_resized, persist=True)
            for result in results:
                boxes = result.boxes.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    class_id = int(box.cls[0])  # klasa boxa

                    if class_id == 2:  # person
                            r = box.xyxy[0].astype(int)
                            x_min, y_min, x_max, y_max = r

                            person_region = frame[y_min:y_max, x_min:x_max]
                            pred = test_predict(person_region)

                            # Aktualizuj dane klasyfikacji dla osoby
                            classifications_dict[id] = pred

            for id, pred in classifications_dict.items():
                try:
                    # Get the coordinates of the person from the detection results
                    box_indices = np.where(ids == id)[0]
                    if len(box_indices) > 0:
                        box_index = box_indices[0]
                        r = boxes[box_index].xyxy[0].astype(int)

                        # Display the activity above the person's head
                        cv2.putText(frame_resized,
                                    f'{pred[0]}',
                                    (r[0] + 20, r[1] + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0),
                                    thickness=2
                                    )

                        cv2.putText(frame_resized,
                                    f'{round(pred[1], 2)}',
                                    (r[0] + 20, r[1] + 85),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255),
                                    thickness=2
                                    )
                    else:
                        print(f"No box found for ID {id}. Skipping processing for this ID.")
                except ValueError as e:
                    # Handle the situation where the ID is not found in the list of ids
                    print(f"ID {id} was not found in the list of ids. Skipping processing for this ID.")

            cv2.imshow('Frame', frame_resized)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
