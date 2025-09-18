import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset which includes 'car', 'truck', 'bus')
model = YOLO("yolov8n.pt")  # small, fast model; use yolov8s.pt or yolov8m.pt for better accuracy

# Video
cap = cv2.VideoCapture("car.mp4")

# Counting variables
car_count = 0
line_y = 400   # y position of counting line
offset = 15    # tolerance
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  
    # classes: 2=car, 3=motorbike, 5=bus, 7=truck

    # Draw line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)

    if results[0].boxes.id is not None:  # tracked objects
        for box, obj_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            obj_id = int(obj_id)

            # Draw box & ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Count vehicles crossing the line
            if (line_y - offset) < cy < (line_y + offset):
                if obj_id not in counted_ids:
                    car_count += 1
                    counted_ids.add(obj_id)

    # Show count
    cv2.putText(frame, f"Cars: {car_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO Car Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == 13:  # ESC or Enter
        break
print(car_count)    

cap.release()
cv2.destroyAllWindows()