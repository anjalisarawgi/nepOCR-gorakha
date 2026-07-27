from surya.detection import DetectionPredictor
from PIL import Image, ImageDraw


predictor = DetectionPredictor()
image = Image.open("images/K 474_00000243.jpeg").convert("RGB")
results = predictor([image])

# Draw boxes
draw = ImageDraw.Draw(image)
for bbox in results[0].bboxes:
    box = bbox.bbox  # [x1, y1, x2, y2]
    draw.rectangle(box, outline="red", width=2)

image.save("K 474_00000243_surya.jpeg")
print(f"Done! {len(results[0].bboxes)} regions found.")