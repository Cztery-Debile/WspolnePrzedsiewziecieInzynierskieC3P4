import random
from collections import defaultdict
import cv2
from ultralytics import YOLO
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

# load yolov8 model
model_yolo = YOLO('../models/best_today.pt')
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification('czynnosci_model.pt')


center_dict = defaultdict(list)
object_id_list = []
color_dict = {}
max_points = 20  # maksymalna liczba punktów śledzenia

# load video
video_path = 'x.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

def random_color():
    return tuple(int(random.random() * 255) for _ in range(3))

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
                        if id in classifications_dict and frame_count % 10 == 0:
                            classification_text = classifications_dict[id]
                            persisted_classifications[id] = classification_text

        for id, classification_text in persisted_classifications.items():
            if id in classifications_dict:
                r = boxes[ids.tolist().index(id)].xyxy[0].astype(int)
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
                            0.5, (255, 255, 255),
                            thickness=2
                            )

        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()