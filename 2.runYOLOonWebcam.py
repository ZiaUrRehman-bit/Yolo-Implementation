from ultralytics import YOLO
import cv2 as cv

cap = cv.VideoCapture(0)

# 1. create the model, by giving weights, (nano, medium, large), just type the name, it will download the weights
# yolov8n.pt "yolo version 8 n means nano version"
model = YOLO("../YOLO-Weights/yolov8n.pt")

while True:
    
    Success, frame = cap.read()

    results = model.predict(frame)
    
    result = results[0]
    if result:
        print(result.boxes[0].cls)
        print(result.boxes[0].conf)
        print(result.boxes[0].xywh)

    cv.imshow("webcam", frame)
    # cv.imshow("webcam", frame)
    
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    
cv.destroyAllWindows()