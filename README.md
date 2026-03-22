# 🚗 Yawn Detection System — ADAS Module

> **Part of an Advanced Driver Assistance System (ADAS)** — A real-time driver drowsiness detection module that monitors yawning frequency and triggers alerts to prevent road accidents.

---

## 📌 Project Overview

This module detects driver yawning in real time using a webcam feed and classifies the driver's alertness into three zones based on yawn frequency per minute. It is built as a standalone ADAS module and includes **two independent detection approaches**:

| Approach | File | Method |
|---|---|---|
| Mediapipe + OpenCV | `main.py` | Facial landmark-based Mouth Aspect Ratio (MAR) |
| YOLOv8 (Custom Trained) | `yolo_stream1.py`, `yolo_stream2.py` | Deep learning object detection |

---

## 🏗️ Repository Structure

```
Yawn-Detection-System/
│
├── mediapipe_approach/
│   └── main.py                  # Yawn detection using Mediapipe facial landmarks
│
├── yolo_approach/
│   ├── yolo_stream1.py          # YOLO streaming – basic version (conf=0.3)
│   ├── yolo_stream2.py          # YOLO streaming – improved version (conf=0.7, cooldown)
│   └── yolo_test.py             # Quick test script using YOLO on webcam
│
├── models/
│   └── yolo_yawn_model.pt       # ⚠️ Not included — see note below
│
├── assets/
│   └── buzzer.mp3               # Alert sound for drowsiness warning
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Approach 1 — Mediapipe + OpenCV (`main.py`)

This approach uses **Google's Mediapipe Face Mesh** to detect 468 facial landmarks in real time, then computes the **Mouth Aspect Ratio (MAR)** to detect whether a person is yawning.

### How It Works

1. Captures live webcam feed using OpenCV.
2. Passes each frame through Mediapipe's Face Mesh model to get facial landmarks.
3. Extracts 4 key mouth landmarks (indices 13, 14, 78, 308):
   - **Vertical distance**: between upper and lower lip center points
   - **Horizontal distance**: between left and right mouth corners
4. Computes MAR:
   ```
   MAR = (vertical / horizontal) × 100
   ```
5. If `MAR ≥ 70`, a yawn is detected. A 2-second cooldown prevents duplicate counts.
6. Yawn count resets every 60 seconds. Zone is assigned based on count.

### Alert Zone System

| Zone | Condition | Colour |
|---|---|---|
| 🟢 GREEN | ≤ 1 yawn/min | Green |
| 🟡 YELLOW | 2–3 yawns/min | Yellow |
| 🔴 RED | ≥ 4 yawns/min | Red + Buzzer |

### Dependencies

```
opencv-python
mediapipe
playsound
numpy
```

---

## 🤖 Approach 2 — YOLOv8 Custom Model (`yolo_stream1.py`, `yolo_stream2.py`)

This approach uses a **custom-trained YOLOv8 model** (`yolo_yawn_model.pt`) to directly detect the "yawn" class in each video frame using bounding box detection.

### How It Works

1. Loads the custom `.pt` model using the `ultralytics` YOLO library.
2. Each frame is passed to `model.predict()` with a confidence threshold.
3. If a bounding box with label `"yawn"` is detected, the yawn counter increments.
4. Counter resets every 60 seconds.

### Difference Between `yolo_stream1.py` and `yolo_stream2.py`

| Feature | `yolo_stream1.py` | `yolo_stream2.py` |
|---|---|---|
| Confidence threshold | `conf=0.3` | `conf=0.7` |
| Cooldown between detections | ❌ No | ✅ Yes (3 seconds) |
| Beep behaviour | Every frame in RED | Once per RED period |
| RED zone threshold | `> 5` yawns | `> 3` yawns |

`yolo_stream2.py` is the **improved version** — recommended for real-world use.

### Alert Zones

| Zone | `yolo_stream1.py` | `yolo_stream2.py` |
|---|---|---|
| 🟢 GREEN | < 1 yawn/min | < 2 yawns/min |
| 🟡 YELLOW | 1–5 yawns/min | 2–3 yawns/min |
| 🔴 RED | > 5 yawns/min | > 3 yawns/min |

### Dependencies

```
opencv-python
ultralytics
```

> ⚠️ The YOLO scripts originally used a hardcoded Windows path for the model. Update line 6 in both files to use a relative path:
> ```python
> model = YOLO('models/yolo_yawn_model.pt')
> ```

> 📌 **Model Weights Not Included:** The custom trained model file `yolo_yawn_model.pt` is not included in this repository. To use the YOLO approach, you will need to retrain the model using a yawn dataset (available on [Roboflow Universe](https://universe.roboflow.com) or [Kaggle](https://www.kaggle.com)). See the **Retraining the Model** section below.
>
> In the meantime, the **Mediapipe approach (`main.py`) works fully without any model file** and is ready to run out of the box.

---

## 🔁 Retraining the YOLO Model

If you want to use the YOLO-based detection, follow these steps to train your own model:

**1. Get a yawn dataset:**
- Go to [Roboflow Universe](https://universe.roboflow.com) → search `yawn detection` → download in **YOLOv8 format**
- Or search on [Kaggle](https://www.kaggle.com) for yawn datasets

**2. Train the model:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # starts from pretrained YOLOv8 nano base
model.train(data='dataset/data.yaml', epochs=50, imgsz=640)
```

**3. Use the trained weights:**
The best trained model will be saved at `runs/train/weights/best.pt`. Copy it to the `models/` folder:
```bash
cp runs/train/weights/best.pt models/yolo_yawn_model.pt
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NAGASAITRINAYACHALUVADI/YAWN-DETECTION.git
cd YAWN-DETECTION
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Detection

**Mediapipe approach:**
```bash
python mediapipe_approach/main.py
```

**YOLO approach (recommended):**
```bash
python yolo_approach/yolo_stream2.py
```

> Press `ESC` to exit the detection window.

---

## 📦 requirements.txt

```
numpy
opencv-python
mediapipe
playsound
ultralytics
torch
torchvision
pillow
```

---

## 🔬 Comparison: Mediapipe vs YOLO

| Factor | Mediapipe (MAR) | YOLOv8 (Custom) |
|---|---|---|
| Technique | Rule-based geometry | Deep learning detection |
| Speed | Very fast (CPU) | GPU recommended |
| Accuracy | Sensitive to angle/lighting | More robust |
| Custom training needed | ❌ No | ✅ Yes |
| Model file needed | ❌ No | ✅ `yolo_yawn_model.pt` |
| Best for | Low-resource devices | High accuracy scenarios |

---

## 🎯 Context: ADAS Module

This project is one module within a larger **Advanced Driver Assistance System (ADAS)**. ADAS aims to improve vehicle safety by automating and enhancing the driving experience. This yawn detection module contributes to the **Driver Monitoring System (DMS)** component by:

- Detecting drowsiness indicators (yawning)
- Triggering graduated alerts before the driver reaches a dangerous state
- Operating in real time on a standard webcam feed

---

## 👤 Author

**Naga Sai Trinayacha Luvadi**  
B.Tech — [Your Branch]  
[Your College Name]

---

## 📄 License

This project is for educational and academic use.
