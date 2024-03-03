import cv2
from ultralytics import YOLO

# zaladowanie modelu
model = YOLO('best.pt')

# zmienne do określenia pozycji myszki
isDrawing = False
selected_areas = []


def select_area(event, x, y, flags, param):
    global isDrawing, selected_areas, frame

    if event == cv2.EVENT_LBUTTONDOWN:  # określenie startowych wspołrzędnych dla myszki
        isDrawing = True
        selected_areas.append((x, y, x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if isDrawing:
            selected_areas[-1] = (selected_areas[-1][0], selected_areas[-1][1], x, y)

            display_frame = frame.copy()
            for area in selected_areas:
                cv2.rectangle(display_frame, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), 2)
            cv2.imshow("Highlighted Region", display_frame)

    elif event == cv2.EVENT_LBUTTONUP:  # określenie końcowej pozycji po puszczeniu przycisku myszki
        isDrawing = False


def refresh_window(window_name):
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, select_area)


def remove_last_area():
    global selected_areas, frame
    if selected_areas:
        selected_areas.pop()
        display_frame = frame.copy()
        for area in selected_areas:
            cv2.rectangle(display_frame, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), 2)
        cv2.imshow("Highlighted Region", display_frame)

# załadowanie obrazu i pokazanie go
image_path = './test2.jpeg'
frame = cv2.imread(image_path)
cv2.imshow("Highlighted Region", frame)
cv2.setMouseCallback("Highlighted Region", select_area)

detected_results = [] # wyniki detekcji

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):  # Naciśnij 'p' aby wykonać detekcję YOLO na wszystkich zaznaczonych obszarach
        for area in selected_areas:
            cropped_region = frame[min(area[1], area[3]):max(area[1], area[3]), min(area[0], area[2]):max(area[0], area[2])]
            results = model.predict(source=cropped_region, conf=0.5)
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls[0])  # klasa boxa
                    confidence = box.conf.astype(float)
                    if class_id == 0:  # okreslenie klasy dla tego modelu 0->person 1->head
                        r = box.xyxy[0].astype(int)
                        class_name = model.names[class_id]
                        x_center = (r[0] + r[2]) // 2
                        y_center = (r[1] + r[3]) // 2
                        cv2.rectangle(cropped_region, r[:2], r[2:], (0, 255, 0), 2)  # rysowanie na obrazie boxa
                        cv2.putText(cropped_region, f'Name: {class_name}', (x_center, y_center),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(cropped_region, f'{confidence}', (x_center, y_center + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    if class_id == 1:
                        r = box.xyxy[0].astype(int)
                        class_name = model.names[class_id]
                        x_center = (r[0] + r[2]) // 2
                        y_center = (r[1] + r[3]) // 2
                        cv2.rectangle(cropped_region, r[:2], r[2:], (0, 0, 255), 2)  # rysowanie na obrazie boxa
                        cv2.putText(cropped_region, f'Name: {class_name}', (x_center, y_center),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(cropped_region, f'{confidence}', (x_center, y_center + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            refresh_window("Highlighted Region")

    elif key == ord('d'):
        remove_last_area()

