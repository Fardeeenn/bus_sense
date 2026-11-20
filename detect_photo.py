from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="Samdutse/pothole-yolov8", filename="best.pt")
model = YOLO(model_path)

results = model.predict("data/pothole.jpg", conf=0.25)
result = results[0]

for box in result.boxes:
    name = model.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"{name}, {confidence:.0%} sure")

print(f"Total found: {len(result.boxes)}")
result.save("data/pothole_result.jpg")