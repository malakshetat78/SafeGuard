import cv2
from ultralytics import YOLO
from collections import deque
import time
import base64
import asyncio
import websockets
import json
import threading
import requests

# =========================
# Load Model
# =========================
model = YOLO("best.pt")
print("Class names:", model.names)

# =========================
# Config
# =========================
CONF_THRESHOLD = 0.6
ALERT_COOLDOWN = 2  # seconds

history = deque(maxlen=5)
prev_time = 0
last_alert_time = 0

class_map = {
    0: "no_helmet",
    1: "helmet",
    2: "gloves",
    3: "forklift",
    4: "head",
    5: "person"
}

# =========================
# Telegram Config
# =========================
TELEGRAM_TOKEN   = "Token"   
TELEGRAM_CHAT_ID = "1826450543"

def send_telegram_alert(frame):
    try:
        msg = (
            "⛔ *NO HELMET DETECTED!*\n"
            f"🕐 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "⚠️ A worker without a helmet was detected!"
        )
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=5
        )
        _, buffer = cv2.imencode('.jpg', frame)
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": "📸 Violation snapshot"},
            files={"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=5
        )
        print("Telegram alert sent!")
    except Exception as e:
        print(f"Telegram error: {e}")

# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)

connected_clients = set()

def process_frame():
    global prev_time, last_alert_time

    ret, frame = cap.read()
    if not ret:
        return None, None

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, conf=0.25)
    detected_objects = []

    for r in results:
        boxes = r.boxes
        history.append(1 if len(boxes) > 0 else 0)

        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < CONF_THRESHOLD:
                continue

            name = class_map.get(cls, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_objects.append(name)

            color = (0, 0, 255) if name == "no_helmet" else (0, 255, 0)
            label = f"{name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Alert logic
    has_no_helmet = "no_helmet" in detected_objects
    alert_triggered = False

    if has_no_helmet:
        cv2.putText(frame, "WARNING: NO HELMET!", (120, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if time.time() - last_alert_time > ALERT_COOLDOWN:
            print("WARNING: NO HELMET DETECTED!")
            alert_triggered = True
            last_alert_time = time.time()
          
            threading.Thread(target=send_telegram_alert, args=(frame.copy(),), daemon=True).start()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Stability
    stable = sum(history) > 2
    status = "STABLE" if stable else "UNSTABLE"
    color_status = (0, 255, 0) if stable else (0, 0, 255)

    cv2.putText(frame, f"STATUS: {status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

    # Encode frame as JPEG base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpg_base64 = base64.b64encode(buffer).decode('utf-8')

    data = {
        "frame": jpg_base64,
        "fps": int(fps),
        "status": status,
        "detected": detected_objects,
        "has_no_helmet": has_no_helmet,
        "alert": alert_triggered
    }

    return data

async def stream_handler(websocket, path=None):
    print(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        while True:
            data = process_frame()
            if data is None:
                break
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.033)  # ~30fps
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connected_clients.discard(websocket)

async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(stream_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())