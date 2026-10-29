from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2, math, os, time, csv
import tensorflow as tf
import numpy as np
from datetime import datetime
import geocoder

from utils.temporal import sudden_motion
from utils.send_alert import send_sms_alert, send_email_alert   # ✅ ADDED

app = Flask(__name__)

# ================= LOAD MODELS =================
yolo = YOLO("models/yolov8n.pt")
cnn_model = tf.keras.models.load_model("models/accident_cnn.h5")

VEHICLE_CLASSES = [2, 3, 5, 7]

# ================= THRESHOLDS =================
IOU_THR = 0.05
DIST_CLOSE = 80
DIST_FAR = 140
CONFIRM_FRAMES = 3

# ================= LOGGING =================
LOG_FILE = "logs/accident_log.csv"
os.makedirs("logs", exist_ok=True)
os.makedirs("static/accidents", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Severity", "Snapshot", "Location"])

# ================= HELPERS =================
def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (a[2]-a[0])*(a[3]-a[1])
    area2 = (b[2]-b[0])*(b[3]-b[1])
    union = area1 + area2 - inter
    return inter/union if union else 0

def center(box):
    return ((box[0]+box[2])//2, (box[1]+box[3])//2)

def distance(a, b):
    return math.dist(center(a), center(b))

def cnn_accident(frame):
    img = cv2.resize(frame, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    score = cnn_model.predict(img, verbose=0)[0][0]
    return score > 0.45, score

def classify_severity(motion, cnn_score):
    if motion and cnn_score > 0.7:
        return "HIGH"
    elif motion or cnn_score > 0.55:
        return "MEDIUM"
    return "LOW"

def get_location():
    g = geocoder.ip("me")
    if g.ok:
        lat, lng = g.latlng
        return f"https://www.google.com/maps?q={lat},{lng}"
    return "Location unavailable"

def log_accident(severity, snapshot, location):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            severity,
            snapshot,
            location
        ])

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

# ================= VIDEO STREAM =================
def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    prev_boxes = []

    accident_counter = 0
    accident_confirmed = False
    alert_sent = False   # ✅ PREVENT MULTIPLE ALERTS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = yolo(frame)

        curr_boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) in VEHICLE_CLASSES:
                    curr_boxes.append(tuple(map(int, b.xyxy[0])))

        # ---------- PHYSICAL EVENT ----------
        physical_event = False

        if prev_boxes and len(curr_boxes) >= 2:
            for i in range(len(curr_boxes)):
                for j in range(i+1, len(curr_boxes)):
                    curr_iou = compute_iou(curr_boxes[i], curr_boxes[j])
                    curr_dist = distance(curr_boxes[i], curr_boxes[j])
                    prev_iou = compute_iou(prev_boxes[i], prev_boxes[j]) if i < len(prev_boxes) and j < len(prev_boxes) else 0
                    prev_dist = distance(prev_boxes[i], prev_boxes[j]) if i < len(prev_boxes) and j < len(prev_boxes) else 999

                    if (prev_iou < 0.01 and curr_iou > IOU_THR) or \
                       (prev_dist > DIST_FAR and curr_dist < DIST_CLOSE):
                        physical_event = True

        motion = prev_frame is not None and sudden_motion(prev_frame, frame)
        if len(curr_boxes) == 1 and motion:
            physical_event = True

        cnn_ok, cnn_score = cnn_accident(frame)

        if physical_event and (cnn_ok or motion):
            accident_counter += 1
        else:
            accident_counter = 0

        # ---------- CONFIRM ACCIDENT ----------
        if accident_counter >= CONFIRM_FRAMES and not accident_confirmed:
            accident_confirmed = True

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            snapshot = f"static/accidents/accident_{timestamp}.jpg"
            cv2.imwrite(snapshot, frame)

            severity = classify_severity(motion, cnn_score)
            location = get_location()

            # ✅ LOG
            log_accident(severity, snapshot, location)

            # ✅ ALERT (ONCE)
            if severity in ["MEDIUM", "HIGH"] and not alert_sent:
                send_sms_alert(
                    f"🚨 ACCIDENT DETECTED\nSeverity: {severity}\n{location}"
                )

                send_email_alert(
                    subject="🚨 Road Accident Alert",
                    body=f"""
Accident Detected

Time: {timestamp}
Severity: {severity}

Location:
{location}
""",
                    image_path=snapshot
                )
                alert_sent = True

        # ---------- DRAW ----------
        if accident_confirmed:
            severity = classify_severity(motion, cnn_score)

            for b in curr_boxes:
                cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,0,255), 3)

            cv2.rectangle(frame, (0, 310), (640, 360), (0,0,0), -1)
            cv2.putText(frame,
                        f"ACCIDENT DETECTED | SEVERITY: {severity}",
                        (20, 345),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,255), 2)

        prev_boxes = curr_boxes.copy()
        prev_frame = frame.copy()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/predict_video", methods=["POST"])
def predict_video():
    file = request.files["video"]
    path = "static/input_video.mp4"
    file.save(path)
    return render_template("result.html", media_type="stream")

@app.route("/video_stream")
def video_stream():
    return Response(generate_video_frames("static/input_video.mp4"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
def generate_live_frames():
    cap = cv2.VideoCapture(0)   # ✅ Webcam
    prev_frame = None
    prev_boxes = []

    accident_counter = 0
    accident_confirmed = False
    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = yolo(frame)

        curr_boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) in VEHICLE_CLASSES:
                    curr_boxes.append(tuple(map(int, b.xyxy[0])))

        # ---------- PHYSICAL EVENT ----------
        physical_event = False
        if prev_boxes and len(curr_boxes) >= 2:
            for i in range(len(curr_boxes)):
                for j in range(i+1, len(curr_boxes)):
                    if compute_iou(curr_boxes[i], curr_boxes[j]) > IOU_THR or \
                       distance(curr_boxes[i], curr_boxes[j]) < DIST_CLOSE:
                        physical_event = True

        motion = prev_frame is not None and sudden_motion(prev_frame, frame)
        if len(curr_boxes) == 1 and motion:
            physical_event = True

        cnn_ok, cnn_score = cnn_accident(frame)

        if physical_event and (cnn_ok or motion):
            accident_counter += 1
        else:
            accident_counter = 0

        # ---------- CONFIRM ACCIDENT ----------
        if accident_counter >= CONFIRM_FRAMES and not accident_confirmed:
            accident_confirmed = True

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            snapshot = f"static/accidents/live_{timestamp}.jpg"
            cv2.imwrite(snapshot, frame)

            severity = classify_severity(motion, cnn_score)
            location = get_location()

            log_accident(severity, snapshot, location)

            if severity in ["MEDIUM", "HIGH"] and not alert_sent:
                send_sms_alert(
                    f"🚨 LIVE ACCIDENT DETECTED\nSeverity: {severity}\n{location}"
                )
                send_email_alert(
                    subject="🚨 Live Accident Alert",
                    body=f"""
Live Accident Detected

Time: {timestamp}
Severity: {severity}

Location:
{location}
""",
                    image_path=snapshot
                )
                alert_sent = True

        # ---------- DRAW ----------
        if accident_confirmed:
            severity = classify_severity(motion, cnn_score)
            cv2.rectangle(frame, (0, 310), (640, 360), (0,0,0), -1)
            cv2.putText(frame,
                        f"ACCIDENT DETECTED | SEVERITY: {severity}",
                        (20, 345),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,255), 2)

        prev_boxes = curr_boxes.copy()
        prev_frame = frame.copy()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()
def generate_live_frames():
    cap = cv2.VideoCapture(0)   # ✅ Webcam
    prev_frame = None
    prev_boxes = []

    accident_counter = 0
    accident_confirmed = False
    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = yolo(frame)

        curr_boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) in VEHICLE_CLASSES:
                    curr_boxes.append(tuple(map(int, b.xyxy[0])))

        # ---------- PHYSICAL EVENT ----------
        physical_event = False
        if prev_boxes and len(curr_boxes) >= 2:
            for i in range(len(curr_boxes)):
                for j in range(i+1, len(curr_boxes)):
                    if compute_iou(curr_boxes[i], curr_boxes[j]) > IOU_THR or \
                       distance(curr_boxes[i], curr_boxes[j]) < DIST_CLOSE:
                        physical_event = True

        motion = prev_frame is not None and sudden_motion(prev_frame, frame)
        if len(curr_boxes) == 1 and motion:
            physical_event = True

        cnn_ok, cnn_score = cnn_accident(frame)

        if physical_event and (cnn_ok or motion):
            accident_counter += 1
        else:
            accident_counter = 0

        # ---------- CONFIRM ACCIDENT ----------
        if accident_counter >= CONFIRM_FRAMES and not accident_confirmed:
            accident_confirmed = True

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            snapshot = f"static/accidents/live_{timestamp}.jpg"
            cv2.imwrite(snapshot, frame)

            severity = classify_severity(motion, cnn_score)
            location = get_location()

            log_accident(severity, snapshot, location)

            if severity in ["MEDIUM", "HIGH"] and not alert_sent:
                send_sms_alert(
                    f"🚨 LIVE ACCIDENT DETECTED\nSeverity: {severity}\n{location}"
                )
                send_email_alert(
                    subject="🚨 Live Accident Alert",
                    body=f"""
Live Accident Detected

Time: {timestamp}
Severity: {severity}

Location:
{location}
""",
                    image_path=snapshot
                )
                alert_sent = True

        # ---------- DRAW ----------
        if accident_confirmed:
            severity = classify_severity(motion, cnn_score)
            cv2.rectangle(frame, (0, 310), (640, 360), (0,0,0), -1)
            cv2.putText(frame,
                        f"ACCIDENT DETECTED | SEVERITY: {severity}",
                        (20, 345),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,255), 2)

        prev_boxes = curr_boxes.copy()
        prev_frame = frame.copy()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/live")  # ✅ THIS SHOULD BE AT MODULE LEVEL
def live():
    return render_template("result.html", media_type="live")

@app.route("/live_stream")
def live_stream():
    return Response(
        generate_live_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    with open(LOG_FILE, "r") as f:
        accidents = list(csv.DictReader(f))
    return render_template("dashboard.html", accidents=accidents)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
