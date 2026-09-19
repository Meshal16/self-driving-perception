import cv2
import torch
from ultralytics import YOLO


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = "mps"

elif torch.cuda.is_available():
    device = "cuda"

else:
    device = "cpu"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")

print(f"YOLO ready on {device}")


# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(image_input, base_image=None):

    # --------------------------------------------------------
    # 1. YOLO INFERENCE
    # --------------------------------------------------------
    # image_input can be:
    # 1. Image path (string)
    # 2. Video frame (NumPy array)

    results = model(
        image_input,
        device=device,
        verbose=False
    )

    first_result = results[0]


    # --------------------------------------------------------
    # 2. CHOOSE IMAGE TO DRAW ON
    # --------------------------------------------------------

    if base_image is None:

        # Image path
        if isinstance(image_input, str):

            img = cv2.imread(
                image_input
            )

            if img is None:

                raise ValueError(
                    f"Could not read image: {image_input}"
                )

        # Video frame
        else:

            img = image_input.copy()

    else:

        # Draw on segmentation/lane result
        img = base_image.copy()


    # --------------------------------------------------------
    # 3. STORE CAR BOXES
    # --------------------------------------------------------

    car_boxes = []


    # --------------------------------------------------------
    # 4. LOOP THROUGH DETECTIONS
    # --------------------------------------------------------

    for box in first_result.boxes:

        # Class
        class_id = int(
            box.cls[0]
        )

        class_name = first_result.names[
            class_id
        ]


        # Confidence
        confidence = float(
            box.conf[0]
        )


        # Bounding box
        pos = box.xyxy[0]


        x1 = int(
            pos[0]
        )

        y1 = int(
            pos[1]
        )

        x2 = int(
            pos[2]
        )

        y2 = int(
            pos[3]
        )


        # ----------------------------------------------------
        # SAVE CAR BOXES FOR DRIVING LOGIC
        # ----------------------------------------------------

        if class_name == "car":

            car_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )


        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # LABEL SIZE
        # ----------------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )


        # ----------------------------------------------------
        # KEEP LABEL INSIDE IMAGE
        # ----------------------------------------------------

        label_top = max(
            y1 - text_height - 10,
            0
        )


        # ----------------------------------------------------
        # RED LABEL BACKGROUND
        # ----------------------------------------------------

        cv2.rectangle(
            img,
            (
                x1,
                label_top
            ),
            (
                x1 + text_width + 6,
                y1
            ),
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # TEXT POSITION
        # ----------------------------------------------------

        label_y = max(
            y1,
            text_height + 10
        )


        # ----------------------------------------------------
        # DRAW LABEL
        # ----------------------------------------------------

        cv2.putText(
            img,
            label,
            (
                x1,
                label_y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )


    # ========================================================
    # RETURN
    # ========================================================

    return img, car_boxes