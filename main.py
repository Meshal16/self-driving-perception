import cv2
from src.detection import detect_objects


# Video path
video_path = "data/test_videos/road_video.mp4"


# Open video
cap = cv2.VideoCapture(video_path)


# Check if video opened correctly
if not cap.isOpened():
    print("Could not open video")
    exit()


print("Running YOLO on video...")


while True:

    # Read one frame
    ret, frame = cap.read()

    # Video finished
    if not ret:
        break


    # Run YOLO on the current frame
    result, car_boxes = detect_objects(frame)


    # Show result
    cv2.imshow(
        "YOLO Video Detection",
        result
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release video
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()

print("Video finished.")