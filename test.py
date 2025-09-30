from ultralytics import YOLO

# Load your custom model
model = YOLO('C:\\Users\\trina\\OneDrive\\Desktop\\YAWN\\yolo_yawn_model.pt')


# Predict on image
results = model.predict(
    source=0 ,# You can change this to 0 for webcam
    conf=0.3,             # Confidence threshold
    show=True             # Show the result in a window
)

# Optional: Save the results
results[0].save(filename="result.jpg")
