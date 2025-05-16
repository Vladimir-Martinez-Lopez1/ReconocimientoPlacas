import cv2
from ultralytics import YOLO
import time

# Load the model
model_path = 'C:/Users/Vladimir/Documents/concurso/yes/yolo-plate-recognition-main/runs/detect/train2/weights/best.pt'  # Adjust to your model's path
model = YOLO(model_path)

# Open the video file
video_path = 'C:/Users/Vladimir/Documents/concurso/yes/yolo-plate-recognition-main/carro2.mp4'  # Adjust as needed
cap = cv2.VideoCapture(video_path)

# Set target resolution (optional, to reduce processing load)
target_width = 1280
target_height = 768

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize the frame for better processing performance
    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(frame, (target_width, target_height))

    # Start time for measuring FPS
    start_time = time.time()

    # Run YOLO detection on the frame
    results = model.predict(frame, imgsz=(target_width, target_height))

    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('License Plate Detection', annotated_frame)

    # Print FPS (frames per second) for performance monitoring
    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()