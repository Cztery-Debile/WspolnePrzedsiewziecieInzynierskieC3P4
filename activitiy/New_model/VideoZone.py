import random
import cv2
import numpy as np
import pandas as pd
from keras.src.saving import load_model
from ultralytics import YOLO
from PIL import Image

# load yolov8 model
model_yolo = YOLO('../../models/best_today.pt')
cnn_model = load_model('human_model_last.h5')

# load video
video_path = '../x.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

specified_region = (700, 280, 780, 350) # obszar wykrywania

frame_count = 0
classifications_dict = {}
skip_frames = 2


def read_image(fn):
    image = Image.open(fn)
    resized_image = image.resize((224, 224))
    return np.asarray(resized_image)

#predict the class and the confidence of the prediction
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
    return predicted_class,confidence


# read frames
while ret:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

        frame_resized = cv2.resize(frame, (1280, 720))

        cv2.rectangle(frame_resized, (specified_region[0], specified_region[1]), (specified_region[2], specified_region[3]), (0, 255, 0), 2)

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

                        # Sprawdzenie, czy osoba znajduje się w danym obszarze
                        if (x_max >= specified_region[0] and y_max >= specified_region[1] and
                                x_min <= specified_region[2] and y_min <= specified_region[3]):
                            # Wykonaj predykcję dla osoby
                            x = test_predict(frame_resized)

                            # Aktualizuj dane klasyfikacji dla osoby
                            classifications_dict[id] = x

                        else:
                            # Jeśli osoba opuściła obszar, usuń ją z classifications_dict
                            if id in classifications_dict:
                                classifications_dict.pop(id)

            # Wyświetlenie aktualnych czynności dla każdej osoby
            for id, x in classifications_dict.items():
                try:

                    box_indices = np.where(ids == id)[0]
                    if len(box_indices) > 0:
                        box_index = box_indices[0]
                        r = boxes[box_index].xyxy[0].astype(int)

                        cv2.rectangle(frame_resized, r[:2], r[2:], (255,0,0), 2)  # rysowanie na obrazie boxa
                        # Display the activity above the person's head
                        cv2.putText(frame_resized,
                                    f'{x}',
                                    (r[0] + 20, r[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0),
                                    thickness=3
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