def get_car_bottom_center(car_box):

    x1, y1, x2, y2 = car_box

    center_x = (x1 + x2) // 2
    bottom_y = y2

    return center_x, bottom_y
def get_lane_x_at_y(lane_points, target_y):

    closest_point = min(
        lane_points,
        key=lambda point: abs(point[1] - target_y)
    )

    return closest_point[0]
def is_car_in_ego_lane(car_box, left_lane, right_lane):

    car_x, car_y = get_car_bottom_center(car_box)

    left_x = get_lane_x_at_y(
        left_lane,
        car_y
    )

    right_x = get_lane_x_at_y(
        right_lane,
        car_y
    )

    return left_x < car_x < right_x
def find_ego_lanes(lanes, image_width):

    image_center = image_width // 2

    left_candidates = []
    right_candidates = []

    for lane in lanes:

        if len(lane) == 0:
            continue

        # Use the lowest predicted point
        bottom_point = max(
            lane,
            key=lambda point: point[1]
        )

        lane_x = bottom_point[0]

        if lane_x < image_center:
            left_candidates.append(lane)

        else:
            right_candidates.append(lane)


    if not left_candidates or not right_candidates:
        return None, None


    # Closest left lane to camera center
    left_lane = max(
        left_candidates,
        key=lambda lane: max(
            lane,
            key=lambda point: point[1]
        )[0]
    )


    # Closest right lane to camera center
    right_lane = min(
        right_candidates,
        key=lambda lane: max(
            lane,
            key=lambda point: point[1]
        )[0]
    )


    return left_lane, right_lane

def get_car_proximity(car_box, image_height):

    x1, y1, x2, y2 = car_box

    bottom_y = y2

    proximity = bottom_y / image_height

    return proximity

def make_driving_decision(cars_in_ego_lane, image_height):

    # No car ahead
    if len(cars_in_ego_lane) == 0:
        return "CLEAR", 0.0


    # Find the closest car
    closest_proximity = 0.0

    for car_box in cars_in_ego_lane:

        proximity = get_car_proximity(
            car_box,
            image_height
        )

        if proximity > closest_proximity:
            closest_proximity = proximity


    # Simple educational thresholds
    if closest_proximity >= 0.75:
        decision = "STOP"

    elif closest_proximity >= 0.50:
        decision = "SLOW"

    else:
        decision = "CLEAR"


    return decision, closest_proximity