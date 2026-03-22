import cv2
import time
import winsound
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('models/yolo_yawn_model.pt') # Update path if needed

# Initialize webcam
cap = cv2.VideoCapture(0)

# Counter and timer
yawn_count = 0
start_time = time.time()

# Alert thresholds
def get_alert_level(count):
    if count > 5:
        return "RED"
    elif count < 1:
        return "GREEN"
    else:
        return "YELLOW"

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam error")
        break

    # Predict using YOLOv8
    results = model.predict(frame, conf=0.3, verbose=False)
    names = results[0].names
    boxes = results[0].boxes

    # Check if 'yawn' detected in current frame
    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]
        if label.lower() == "yawn":
            yawn_count += 1
            break  # Count only once per frame

    # Check for 60 seconds passed
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:
        yawn_count = 0
        start_time = time.time()

    # Get alert level
    alert_level = get_alert_level(yawn_count)

    # Draw alert level on frame
    color = (0, 255, 0) if alert_level == "GREEN" else (0, 255, 255) if alert_level == "YELLOW" else (0, 0, 255)
    cv2.putText(frame, f'ALERT: {alert_level}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f'Yawn Count: {yawn_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Beep in RED alert
    if alert_level == "RED":
        winsound.Beep(1000, 300)  # frequency, duration in ms

    # Show the frame
    cv2.imshow("Yawn Detection - 3 Level Alert", frame)

    # Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything
cap.release()
cv2.destroyAllWindows()

