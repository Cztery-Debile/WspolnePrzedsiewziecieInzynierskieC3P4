from collections import defaultdict

from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8s.pt')

# load video
video_path = './shop.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for box, id in zip(boxes,ids):
                class_id = int(box.cls[0])  # klasa boxa
                if class_id == 0:   # okreslenie klasy dla tego modelu 0->person 1->head
                    r = box.xyxy[0].astype(int)  #
                    class_name = model.names[class_id]  #
                    confidence = box[0].conf.astype(float)
                    cv2.rectangle(frame, r[:2], r[2:], (0, 255, 0), 2)  # rysowanie na obrazie boxa
                    cv2.putText(frame, f'Name: {id}', (r[0] + 50, r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, f'{confidence}', (r[0] + 10, r[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                if class_id == 1:
                    r = box.xyxy[0].astype(int)
                    class_name = model.names[class_id]  #
                    cv2.rectangle(frame, r[:2], r[2:], (0, 0, 255), 2)  # rysowanie na obrazie boxa
                    cv2.putText(frame, f'Name: {id}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Wyświetlanie przetworzonego obrazu
        cv2.imshow("act", frame)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break