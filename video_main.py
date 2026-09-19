import cv2
import numpy as np
import os

from src.segmentation import segment_road
from src.lane import lane_detection
from src.detection import detect_objects

from src.driving_logic import (
    find_ego_lanes,
    is_car_in_ego_lane,
    make_driving_decision
)


# ============================================================
# VIDEO PATHS
# ============================================================

video_path = "/Users/meshalaziz/Desktop/self_driving_perception/data/test_images/test_videos/Driving02.mp4"

output_folder = "output"
output_path = os.path.join(
    output_folder,
    "final_perception_demo.mp4"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# LANE STABILITY SETTINGS
# ============================================================

SMOOTHING_ALPHA = 0.30
MAX_LANE_JUMP = 150
MAX_MISSING_FRAMES = 5


# ============================================================
# LANE MEMORY
# ============================================================

previous_lanes = [
    [],
    [],
    [],
    []
]

missing_frames = [
    0,
    0,
    0,
    0
]


# ============================================================
# GET REPRESENTATIVE LANE X
# ============================================================

def get_lane_center_x(lane):

    if len(lane) == 0:
        return None

    x_values = [
        point[0]
        for point in lane
    ]

    return int(
        np.mean(x_values)
    )


# ============================================================
# SMOOTH LANE
# ============================================================

def smooth_lane(
    old_lane,
    new_lane,
    alpha=0.30
):

    if len(old_lane) == 0:
        return new_lane

    if len(new_lane) == 0:
        return old_lane

    target_length = min(
        len(old_lane),
        len(new_lane)
    )

    if target_length < 2:
        return new_lane

    old_indices = np.linspace(
        0,
        len(old_lane) - 1,
        target_length
    ).astype(int)

    new_indices = np.linspace(
        0,
        len(new_lane) - 1,
        target_length
    ).astype(int)

    smoothed_lane = []

    for old_index, new_index in zip(
        old_indices,
        new_indices
    ):

        old_x, old_y = old_lane[
            old_index
        ]

        new_x, new_y = new_lane[
            new_index
        ]

        smooth_x = int(
            alpha * new_x
            +
            (1 - alpha) * old_x
        )

        smooth_y = int(
            alpha * new_y
            +
            (1 - alpha) * old_y
        )

        smoothed_lane.append(
            (
                smooth_x,
                smooth_y
            )
        )

    return smoothed_lane


# ============================================================
# STABILIZE LANES
# ============================================================

def stabilize_lanes(
    new_lanes,
    previous_lanes,
    missing_frames
):

    stabilized = []

    for lane_index in range(4):

        new_lane = new_lanes[
            lane_index
        ]

        old_lane = previous_lanes[
            lane_index
        ]

        # ----------------------------------------------------
        # NEW LANE DETECTED
        # ----------------------------------------------------

        if len(new_lane) >= 2:

            new_center = get_lane_center_x(
                new_lane
            )

            old_center = get_lane_center_x(
                old_lane
            )

            # Reject large sudden jumps
            if (
                old_center is not None
                and
                abs(
                    new_center - old_center
                ) > MAX_LANE_JUMP
            ):

                if (
                    missing_frames[lane_index]
                    <
                    MAX_MISSING_FRAMES
                ):

                    stabilized.append(
                        old_lane
                    )

                    missing_frames[
                        lane_index
                    ] += 1

                    continue

            # Smooth valid prediction
            smoothed = smooth_lane(
                old_lane,
                new_lane,
                SMOOTHING_ALPHA
            )

            stabilized.append(
                smoothed
            )

            missing_frames[
                lane_index
            ] = 0

        # ----------------------------------------------------
        # LANE MISSING
        # ----------------------------------------------------

        else:

            if (
                len(old_lane) >= 2
                and
                missing_frames[lane_index]
                <
                MAX_MISSING_FRAMES
            ):

                stabilized.append(
                    old_lane
                )

                missing_frames[
                    lane_index
                ] += 1

            else:

                stabilized.append(
                    []
                )

                missing_frames[
                    lane_index
                ] = 0

    return stabilized


# ============================================================
# DRAW STABILIZED LANES
# ============================================================

def draw_stabilized_lanes(
    image,
    lanes
):

    result = image.copy()

    for lane in lanes:

        if len(lane) < 2:
            continue

        points = np.array(
            lane,
            dtype=np.int32
        )

        cv2.polylines(
            result,
            [points],
            False,
            (255, 0, 0),
            4
        )

        for point in lane:

            cv2.circle(
                result,
                point,
                3,
                (0, 255, 255),
                -1
            )

    return result


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud(
    image,
    total_cars,
    ego_cars,
    proximity,
    decision
):

    result = image.copy()

    # HUD BACKGROUND
    overlay = result.copy()

    cv2.rectangle(
        overlay,
        (20, 20),
        (330, 180),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        result,
        0.35,
        0,
        result
    )

    # TITLE
    cv2.putText(
        result,
        "PERCEPTION STATUS",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # VEHICLES
    cv2.putText(
        result,
        f"Vehicles: {total_cars}",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # EGO LANE VEHICLES
    cv2.putText(
        result,
        f"In Ego Lane: {ego_cars}",
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # PROXIMITY
    cv2.putText(
        result,
        f"Proximity: {proximity:.2f}",
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # DECISION COLOR
    if decision == "CLEAR":

        decision_color = (
            0,
            255,
            0
        )

    elif decision == "SLOW":

        decision_color = (
            0,
            255,
            255
        )

    else:

        decision_color = (
            0,
            0,
            255
        )

    # DECISION
    cv2.putText(
        result,
        f"Decision: {decision}",
        (40, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        decision_color,
        2
    )

    return result


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    print(
        "Could not open video"
    )

    exit()


# ============================================================
# GET ORIGINAL VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


print(
    f"Video FPS: {fps}"
)

print(
    f"Video Size: {width}x{height}"
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (
        width,
        height
    )
)


if not writer.isOpened():

    print(
        "Could not create output video"
    )

    cap.release()

    exit()


print(
    "Running Autonomous Driving Perception..."
)


# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]


    # ========================================================
    # 1. ROAD SEGMENTATION
    # ========================================================

    segmentation_result = segment_road(
        frame
    )


    # ========================================================
    # 2. RAW LANE DETECTION
    # ========================================================

    _, raw_lanes = lane_detection(
        frame
    )


    # ========================================================
    # 3. STABILIZE LANES
    # ========================================================

    stable_lanes = stabilize_lanes(
        raw_lanes,
        previous_lanes,
        missing_frames
    )


    previous_lanes = [
        lane.copy()
        for lane in stable_lanes
    ]


    # ========================================================
    # 4. DRAW LANES
    # ========================================================

    lane_result = draw_stabilized_lanes(
        segmentation_result,
        stable_lanes
    )


    # ========================================================
    # 5. YOLO
    # ========================================================

    detection_result, car_boxes = detect_objects(
        frame,
        base_image=lane_result
    )


    # ========================================================
    # 6. FIND EGO LANES
    # ========================================================

    left_lane, right_lane = find_ego_lanes(
        stable_lanes,
        width
    )


    # ========================================================
    # 7. FIND CARS INSIDE EGO LANE
    # ========================================================

    cars_in_ego_lane = []

    if (
        left_lane is not None
        and
        right_lane is not None
    ):

        for car_box in car_boxes:

            if is_car_in_ego_lane(
                car_box,
                left_lane,
                right_lane
            ):

                cars_in_ego_lane.append(
                    car_box
                )


    # ========================================================
    # 8. DRIVING DECISION
    # ========================================================

    decision, proximity = make_driving_decision(
        cars_in_ego_lane,
        height
    )


    # ========================================================
    # 9. HUD
    # ========================================================

    final_result = draw_hud(
        detection_result,
        len(car_boxes),
        len(cars_in_ego_lane),
        proximity,
        decision
    )


    # ========================================================
    # 10. SAVE FRAME TO FINAL VIDEO
    # ========================================================

    writer.write(
        final_result
    )


    # ========================================================
    # 11. DISPLAY
    # ========================================================

    cv2.imshow(
        "Autonomous Driving - Dynamic Perception",
        final_result
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


print(
    "Video finished."
)

print(
    f"Final video saved to: {output_path}"
)