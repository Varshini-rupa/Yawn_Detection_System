import cv2
import mediapipe as mp
import time
from playsound import playsound
import threading
import math

# Function to play beep in a separate thread
def play_beep():
    threading.Thread(target=playsound, args=("buzzer.mp3",), daemon=True).start()

# Calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks, image_width, image_height):
    def distance(p1, p2):
        x1, y1 = int(landmarks[p1].x * image_width), int(landmarks[p1].y * image_height)
        x2, y2 = int(landmarks[p2].x * image_width), int(landmarks[p2].y * image_height)
        return math.hypot(x2 - x1, y2 - y1)

    vertical = distance(13, 14)
    horizontal = distance(78, 308)
    mar = (vertical / horizontal) * 100  # convert to percentage
    return mar

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# OpenCV video
cap = cv2.VideoCapture(0)
yawn_times = []
last_yawn_time = 0
cooldown_seconds = 2  # don't detect new yawn within 2s

print("▶ Starting Yawn Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    image_height, image_width = frame.shape[:2]
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw lip landmarks
            for idx in [13, 14, 78, 308]:
                x = int(face_landmarks.landmark[idx].x * image_width)
                y = int(face_landmarks.landmark[idx].y * image_height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            mar = calculate_mar(face_landmarks.landmark, image_width, image_height)

            if mar >= 70:
                if current_time - last_yawn_time > cooldown_seconds:
                    yawn_times.append(current_time)
                    last_yawn_time = current_time
                    play_beep()
                    cv2.putText(frame, "Yawning!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Filter out old yawns (older than 60s)
    yawn_times = [t for t in yawn_times if current_time - t <= 60]
    yawn_count = len(yawn_times)

    # Debug
    print(f"[DEBUG] Yawns in last 60s: {yawn_count}")

    # Show count and zone
    cv2.putText(frame, f"Yawns in 60s: {yawn_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if yawn_count <= 1:
        zone = "GREEN ZONE"
        color = (0, 255, 0)
    elif 2 <= yawn_count <= 3:
        zone = "YELLOW ZONE"
        color = (0, 255, 255)
    else:
        zone = "RED ZONE"
        color = (0, 0, 255)

    cv2.putText(frame, f"Zone: {zone}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
