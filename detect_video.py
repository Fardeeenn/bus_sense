import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="Samdutse/pothole-yolov8", filename="best.pt")
model = YOLO(model_path)

video = cv2.VideoCapture("data/road_video.mp4")
if not video.isOpened():
    print("Could not open video. Check the file name and path.")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    "data/road_vid_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)
frame_number = 0

while True:
    success, frame = video.read()
    if not success:
        break

    result = model.predict(frame, conf=0.25, verbose=False)[0]
    seconds = frame_number / fps

    for box in result.boxes:
        confidence = float(box.conf[0])
        print(f"{seconds:.1f}s: pothole, {confidence:.0%} sure")

    writer.write(result.plot())
    frame_number += 1

video.release()
writer.release()
print(f"Done. Frames checked: {frame_number}")