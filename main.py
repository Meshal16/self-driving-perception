from src.detection import detect_objects
from src.segmentation import segment_road
from src.lane import lane_detection

from src.driving_logic import (
    find_ego_lanes,
    is_car_in_ego_lane,
    make_driving_decision
)

import cv2


# ==========================================
# IMAGE
# ==========================================

image_path = "data/test_images/road_03.png"


# ==========================================
# 1. ROAD SEGMENTATION
# ==========================================

print("Running road segmentation...")

segmentation_result = segment_road(
    image_path
)


# ==========================================
# 2. AI LANE DETECTION
# ==========================================

print("Running UFLDv2 lane detection...")

lane_result, lanes = lane_detection(
    image_path,
    base_image=segmentation_result
)

print(
    "Detected lanes:",
    [len(lane) for lane in lanes]
)


# ==========================================
# 3. OBJECT DETECTION
# ==========================================

print("Running YOLO object detection...")

final_result, car_boxes = detect_objects(
    image_path,
    base_image=lane_result
)

print(
    "Cars detected:",
    len(car_boxes)
)


# ==========================================
# 4. FIND EGO LANE
# ==========================================

height, width = final_result.shape[:2]

left_lane, right_lane = find_ego_lanes(
    lanes,
    width
)


# ==========================================
# 5. FIND CARS INSIDE EGO LANE
# ==========================================

cars_in_ego_lane = []

if left_lane is not None and right_lane is not None:

    for car_box in car_boxes:

        if is_car_in_ego_lane(
            car_box,
            left_lane,
            right_lane
        ):

            cars_in_ego_lane.append(
                car_box
            )


print(
    "Cars in ego lane:",
    len(cars_in_ego_lane)
)


# ==========================================
# 6. DRIVING DECISION
# ==========================================

decision, proximity = make_driving_decision(
    cars_in_ego_lane,
    height
)

print(
    "Closest car proximity:",
    round(proximity, 2)
)

print(
    "Driving Decision:",
    decision
)


# ==========================================
# 7. PERCEPTION HUD
# ==========================================

# Create copy for transparent HUD
overlay = final_result.copy()


# Black HUD background
cv2.rectangle(
    overlay,
    (25, 25),
    (330, 185),
    (0, 0, 0),
    -1
)


# Make HUD transparent
final_result = cv2.addWeighted(
    overlay,
    0.65,
    final_result,
    0.35,
    0
)


# ------------------------------------------
# HUD TITLE
# ------------------------------------------

cv2.putText(
    final_result,
    "PERCEPTION STATUS",
    (45, 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)


# ------------------------------------------
# VEHICLES
# ------------------------------------------

cv2.putText(
    final_result,
    f"Vehicles: {len(car_boxes)}",
    (45, 85),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    1
)


# ------------------------------------------
# VEHICLES IN EGO LANE
# ------------------------------------------

cv2.putText(
    final_result,
    f"In Ego Lane: {len(cars_in_ego_lane)}",
    (45, 110),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    1
)


# ------------------------------------------
# PROXIMITY
# ------------------------------------------

cv2.putText(
    final_result,
    f"Proximity: {proximity:.2f}",
    (45, 135),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    1
)


# ==========================================
# DECISION COLOR
# ==========================================

if decision == "STOP":

    decision_color = (0, 0, 255)


elif decision == "SLOW":

    decision_color = (0, 165, 255)


else:

    decision_color = (0, 255, 0)


# ==========================================
# DISPLAY DECISION
# ==========================================

cv2.putText(
    final_result,
    f"Decision: {decision}",
    (45, 165),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    decision_color,
    2
)


# ==========================================
# 8. FINAL PERCEPTION
# ==========================================

cv2.imshow(
    "Autonomous Driving - Final Perception",
    final_result
)

cv2.waitKey(0)

cv2.destroyAllWindows()