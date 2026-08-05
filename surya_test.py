from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from PIL import Image, ImageDraw, ImageFont
from surya.inference import SuryaInferenceManager


# Initialize predictors
detection_predictor = DetectionPredictor()
manager = SuryaInferenceManager()
recognition_predictor = RecognitionPredictor(manager)

image = Image.open("images/K 175_00000371.jpeg").convert("RGB")


det_results = detection_predictor([image])
rec_results = recognition_predictor([image])

# Draw boxes + recognized text
draw = ImageDraw.Draw(image)
for line in rec_results[0].text_lines:
    box = line.bbox  # [x1, y1, x2, y2]
    text = line.text
    draw.rectangle(box, outline="red", width=2)
    # Draw text above the box
    draw.text((box[0], box[1] - 12), text, fill="blue")

image.save("K 175_00000371_ocr.jpeg")

# Print all recognized text
print(f"Found {len(rec_results[0].text_lines)} lines:\n")
for line in rec_results[0].text_lines:
    print(f"  [{line.confidence:.2f}] {line.text}")