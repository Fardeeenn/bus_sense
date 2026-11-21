import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

MIN_FRAMES = 5

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
    "data/road_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

frame_number = 0
best = {}

while True:
    success, frame = video.read()
    if not success:
        break

    result = model.track(
        frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False
    )[0]
    seconds = frame_number / fps

    for box in result.boxes:
        if box.id is None:
            continue
        track_id = int(box.id[0])
        confidence = float(box.conf[0])
        if track_id not in best:
            best[track_id] = {"confidence": confidence, "seconds": seconds, "frames": 1}
        else:
            best[track_id]["frames"] += 1
            if confidence > best[track_id]["confidence"]:
                best[track_id]["confidence"] = confidence
                best[track_id]["seconds"] = seconds

    writer.write(result.plot())
    frame_number += 1

video.release()
writer.release()

real = {i: info for i, info in best.items() if info["frames"] >= MIN_FRAMES}
print(f"Frames checked: {frame_number}")
print(f"Raw IDs created: {len(best)}")
print(f"Potholes seen in {MIN_FRAMES}+ frames: {len(real)}")
for track_id, info in sorted(real.items()):
    print(f"Pothole #{track_id}: best {info['confidence']:.0%} at {info['seconds']:.1f}s ({info['frames']} frames)")