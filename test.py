#pip install ultralytics opencv-python
"""if not working try: python create venv 
run  .\.venv\Scripts\activate
then pip install ultralytics opencv-python
"""
import cv2
import time
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 1. Initialize YOLO26s model
# The 's' variant is the smallest and fastest.
model = YOLO("yolo26s.pt").to(device)

# 2. Setup Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set resolution for better performance (Optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

print("Starting real-time detection... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Run Inference
    # 'stream=True' is more memory efficient for video processing
    results = model(frame, stream=True)

    for result in results:
        # 4. Use the built-in plot() method for high-performance visualization
        # This draws bounding boxes, labels, and confidence scores automatically
        annotated_frame = result.plot()

        # 5. Calculate and Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 6. Display the Output
        cv2.imshow("YOLO26s Real-Time Detection", annotated_frame)


    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()

cv2.destroyAllWindows()
