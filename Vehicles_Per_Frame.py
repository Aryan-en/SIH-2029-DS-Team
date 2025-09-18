import cv2
from ultralytics import YOLO

# Load YOLOv8 model (make sure you have yolov8s.pt downloaded or use yolov8n.pt for faster)
model = YOLO("yolov8s.pt")  

# Open video file or webcam
cap = cv2.VideoCapture("car.mp4")  # replace with 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    car_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])  # confidence
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                car_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[cls]} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show car count on frame
    cv2.putText(frame, f"Vehicles in Frame: {car_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Car Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()