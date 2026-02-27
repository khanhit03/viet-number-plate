# detector.py - Phần AI phát hiện biển số
from ultralytics import YOLO
import cv2

# Tải mô hình YOLOv8 nano (nhẹ, nhanh)
model = YOLO("yolov8n.pt")  # Model chung, phát hiện "car", "motorcycle", "truck" – biển số thường gắn trên xe

def detect_plate(image_path: str):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        return None, "Không đọc được ảnh!"

    # Dự đoán
    results = model(img)

    # Lọc kết quả xe (class 2: car, 3: motorcycle, 7: truck)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 7]:  # Xe máy, ô tô, xe tải
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"Xe ({conf:.2f})"
                detections.append((label, (x1, y1, x2, y2)))

    if not detections:
        return img, "Không phát hiện xe!"

    # Vẽ khung
    for label, (x1, y1, x2, y2) in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img, f"Phát hiện {len(detections)} xe!"