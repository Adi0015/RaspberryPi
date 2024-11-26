import cv2
import numpy as np
from PIL import Image
import board
import digitalio
import adafruit_rgb_display.st7735 as st7735
from ultralytics import YOLO

# Setup for TFT display
spi = board.SPI()
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D24)
display = st7735.ST7735R(spi, cs=cs_pin, dc=dc_pin, rst=reset_pin, width=128, height=160)

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the TFT resolution
    frame = cv2.resize(frame, (128, 160))

    # Run YOLO inference on the frame
    results = model(frame)

    # Process detections and draw bounding boxes
    for result in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = result.cpu().numpy()
        if class_id == 0:  # Person
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Convert the frame to RGB (Pillow uses RGB, OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    image = Image.fromarray(frame_rgb)

    # Display on the TFT screen
    display.image(image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
