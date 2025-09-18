from flask import Flask, render_template, request, jsonify
import cv2
import os
import math
from ultralytics import YOLO
import threading
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store results for each video
video_results = {}

# Threaded video processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    model = YOLO("yolov8n.pt")

    vehicle_data = {}
    calibration_factor = 0.05
    frame_idx = 0
    car_counts_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_car_count = 0

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

        if results[0].boxes.id is not None:
            for box, obj_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                obj_id = int(obj_id)
                frame_car_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # speed calculation
                if obj_id in vehicle_data:
                    prev_pos = vehicle_data[obj_id]["last_pos"]
                    prev_frame = vehicle_data[obj_id]["last_frame"]
                    dx, dy = cx - prev_pos[0], cy - prev_pos[1]
                    displacement = math.sqrt(dx ** 2 + dy ** 2)
                    dt = (frame_idx - prev_frame) / fps
                    if dt > 0:
                        speed = displacement * calibration_factor / dt
                        vehicle_data[obj_id]["speeds"].append(speed)
                    vehicle_data[obj_id]["last_pos"] = (cx, cy)
                    vehicle_data[obj_id]["last_frame"] = frame_idx
                else:
                    vehicle_data[obj_id] = {"last_pos": (cx, cy), "last_frame": frame_idx, "speeds": []}

        car_counts_per_frame.append(frame_car_count)

        # Average so far
        avg_so_far = sum(car_counts_per_frame) / len(car_counts_per_frame)

        # Encode frame as JPEG and store for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        video_results[video_path]["latest_frame"] = base64.b64encode(frame_bytes).decode('utf-8')
        video_results[video_path]["latest_count"] = frame_car_count
        video_results[video_path]["avg_so_far"] = avg_so_far   # ✅ store avg so far

    # Final results
    avg_car_count = sum(car_counts_per_frame) / len(car_counts_per_frame) if car_counts_per_frame else 0
    all_speeds = [s for v in vehicle_data.values() for s in v["speeds"]]
    avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0

    video_results[video_path]["average_car_count"] = avg_car_count
    video_results[video_path]["average_vehicle_speed"] = avg_speed
    video_results[video_path]["status"] = "done"
    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    file = request.files["video"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Initialize entry for this video
    video_results[path] = {"status": "processing", "latest_frame": "", "latest_count": 0, "avg_so_far": 0}

    # Start processing thread
    threading.Thread(target=process_video, args=(path,)).start()

    return jsonify({"videoPath": path})


@app.route("/video_frame")
def video_frame():
    video_path = request.args.get("video")
    if not video_path or video_path not in video_results:
        return "", 404
    frame = video_results[video_path].get("latest_frame", "")
    count = video_results[video_path].get("latest_count", 0)
    avg_so_far = video_results[video_path].get("avg_so_far", 0)
    if not frame:
        return "", 204
    return jsonify({"frame": frame, "count": count, "avg_so_far": avg_so_far})  # ✅ return both live + avg


@app.route("/final_result")
def final_result():
    video_path = request.args.get("video")
    if not video_path or video_path not in video_results:
        return jsonify({"status": "error"})
    data = video_results[video_path]
    if data["status"] == "done":
        return jsonify(data)
    else:
        return jsonify({"status": "processing"})


if __name__ == "__main__":
    app.run(debug=True)
