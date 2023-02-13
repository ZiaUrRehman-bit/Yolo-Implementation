from ultralytics import YOLO
import cv2 as cv

cap = cv.VideoCapture(0)

# 1. create the model, by giving weights, (nano, medium, large), just type the name, it will download the weights
# yolov8n.pt "yolo version 8 n means nano version"
model = YOLO("../YOLO-Weights/yolov8n.pt")

while True:
    
    Success, frame = cap.read()

    results = model(frame, show=True)
    # cv.imshow("webcam", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    
cv.destroyAllWindows()