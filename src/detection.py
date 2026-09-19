import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def detect_objects(image_path, base_image=None):

    # YOLO inference
    results = model(image_path)
    first_result = results[0]

    # Read original image
    if base_image is None:
        img = cv2.imread(image_path)
    else:
        img = base_image.copy()

    car_boxes = []
    
    for box in first_result.boxes:

        
        class_id = int(box.cls[0])
        class_name = first_result.names[class_id]

        
        confidence = float(box.conf[0])

        
        pos = box.xyxy[0]

        x1 = int(pos[0])
        y1 = int(pos[1])
        x2 = int(pos[2])
        y2 = int(pos[3])
        if class_name == "car":
            car_boxes.append((x1, y1, x2, y2))
        
        label = f"{class_name} {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        2
        )  
        
        cv2.rectangle(
        img,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 6, y1),
        (0, 0, 255),
        -1
        )
        label_y = max(y1, text_height + 10)
        cv2.putText(
            img,
            label,
            (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return img, car_boxes