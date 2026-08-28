# from paddleocr import PaddleOCR
# import cv2

# ocr = PaddleOCR(
#     use_doc_orientation_classify=False, # dont autorotate the whole document 
#     use_doc_unwarping=False,  # dont autocorrect the curvature
#     # use_textline_orientation=False,
#     # engine="paddle", 
#     lang = "ne"
#     )

# result = ocr.predict("images/gorkhapatra_images/nb parishad 12jun1953.jpg")
# # for res in result:
# #     res.print()
# #     res.save_to_img("output")
# #     # res.save_to_json("output")



# for res in result:
#     img = cv2.imread(res['input_path'])
#     polys = res['dt_polys']  # detected text-region polygons

#     for poly in polys:
#         pts = poly.astype(int).reshape(-1, 1, 2)
#         cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#     cv2.imwrite("output/boxes_only_2.jpg", img)

# # ##############################################

# # # from paddleocr import PaddleOCRVL

# # # pipeline = PaddleOCRVL(engine="transformers")  

# # # output = pipeline.predict("images/CA__The Cambridge Manuscript_Page_008.png")

# # # for res in output:
# # #     res.print()
# # #     res.save_to_json(save_path="output")
# # #     res.save_to_img(save_path="output")




# ###########################################################






# import cv2
# import numpy as np
# from lxml import etree
# from datetime import datetime, timezone


# def save_alto_xml(polys, img_shape, image_filename, out_path):
#     h, w = img_shape[:2]

#     NSMAP = {None: "http://www.loc.gov/standards/alto/ns-v4#"}
#     root = etree.Element("alto", nsmap=NSMAP)

#     # --- Description ---
#     description = etree.SubElement(root, "Description")
#     etree.SubElement(description, "MeasurementUnit").text = "pixel"
#     src_info = etree.SubElement(description, "sourceImageInformation")
#     etree.SubElement(src_info, "fileName").text = image_filename
#     ocr_proc = etree.SubElement(description, "OCRProcessing", ID="OCR_1")
#     proc_step = etree.SubElement(ocr_proc, "ocrProcessingStep")
#     etree.SubElement(proc_step, "processingDateTime").text = (
#         datetime.now(timezone.utc).isoformat()
#     )
#     proc_soft = etree.SubElement(proc_step, "processingSoftware")
#     etree.SubElement(proc_soft, "softwareName").text = "PaddleOCR-TextDetection"

#     # --- Layout ---
#     layout = etree.SubElement(root, "Layout")
#     page = etree.SubElement(
#         layout, "Page", ID="page_1", PHYSICAL_IMG_NR="1",
#         WIDTH=str(w), HEIGHT=str(h)
#     )
#     print_space = etree.SubElement(page, "PrintSpace",
#                                     HPOS="0", VPOS="0", WIDTH=str(w), HEIGHT=str(h))

#     for i, poly in enumerate(polys):
#         pts = np.asarray(poly).reshape(-1, 2)
#         x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
#         x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

#         block = etree.SubElement(
#             print_space, "TextBlock", ID=f"block_{i}",
#             HPOS=str(int(x_min)), VPOS=str(int(y_min)),
#             WIDTH=str(int(x_max - x_min)), HEIGHT=str(int(y_max - y_min)),
#         )
#         line = etree.SubElement(
#             block, "TextLine", ID=f"line_{i}",
#             HPOS=str(int(x_min)), VPOS=str(int(y_min)),
#             WIDTH=str(int(x_max - x_min)), HEIGHT=str(int(y_max - y_min)),
#         )
#         shape = etree.SubElement(line, "Shape")
#         points_str = " ".join(f"{int(x)},{int(y)}" for x, y in pts)
#         etree.SubElement(shape, "Polygon", POINTS=points_str)

#         # No recognized text available (detection-only) -> CONTENT left empty
#         etree.SubElement(line, "String", ID=f"string_{i}",
#                           HPOS=str(int(x_min)), VPOS=str(int(y_min)),
#                           WIDTH=str(int(x_max - x_min)), HEIGHT=str(int(y_max - y_min)),
#                           CONTENT="")

#     tree = etree.ElementTree(root)
#     tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


# for res in result:
#     img = cv2.imread(res['input_path'])
#     polys = res['dt_polys']  # detected text-region polygons

#     for poly in polys:
#         pts = poly.astype(int).reshape(-1, 1, 2)
#         cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#     cv2.imwrite("output/boxes_only.jpg", img)
#     save_alto_xml(
#         polys=polys,
#         img_shape=img.shape,
#         image_filename="Gorkhapatra_masthead_9Jan33.jpg",
#         out_path="output/boxes_only_4_nbparishad12jun1953.xml",
#     )


import os
import cv2
import torch
import numpy as np
from PIL import Image
from lxml import etree
from datetime import datetime, timezone
from paddleocr import PaddleOCR
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# detection (lines)
det_model = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, lang="ne",)

# prediction (transcriptions)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
qwen_model.eval()

# QWEN_OCR_PROMPT = ("Transcribe in this image.")
    # "as written. Output only the transcription, with no extra commentary, "
    # "translation, or explanation."



def predict_text(pil_img):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": "Transcribe in this image."},
        ],
    }]
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)

    trimmed_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    result = qwen_processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return result.strip()


# ---------------------------------------------------------------------------
# Crop + ordering helpers
# ---------------------------------------------------------------------------
def get_rotate_crop_image(img, poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC)

    if crop.shape[0] > crop.shape[1] * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def sort_polys_reading_order(polys):
    def centroid(poly):
        pts = np.asarray(poly).reshape(-1, 2)
        return pts[:, 1].mean(), pts[:, 0].mean()  # (y, x)
    return sorted(polys, key=centroid)


# ---------------------------------------------------------------------------
# ALTO writer (now embeds recognized text)
# ---------------------------------------------------------------------------
def _el(parent, tag, **attrs):
    attrs = {k: str(v) for k, v in attrs.items() if v is not None}
    return etree.SubElement(parent, tag, **attrs)


def save_alto_xml(polys, texts, img_shape, image_filename, out_path):
    h, w = img_shape[:2]
    root = etree.Element("alto", nsmap={None: "http://www.loc.gov/standards/alto/ns-v4#"})

    desc = _el(root, "Description")
    _el(desc, "MeasurementUnit").text = "pixel"
    _el(_el(desc, "sourceImageInformation"), "fileName").text = image_filename
    step = _el(_el(desc, "OCRProcessing", ID="OCR_1"), "ocrProcessingStep")
    _el(step, "processingDateTime").text = datetime.now(timezone.utc).isoformat()
    _el(_el(step, "processingSoftware"), "softwareName").text = "PaddleOCR-Det+Qwen2.5-VL-Rec"

    page = _el(_el(root, "Layout"), "Page", ID="page_1", PHYSICAL_IMG_NR="1", WIDTH=w, HEIGHT=h)
    print_space = _el(page, "PrintSpace", HPOS=0, VPOS=0, WIDTH=w, HEIGHT=h)

    for i, (poly, text) in enumerate(zip(polys, texts)):
        pts = np.asarray(poly).reshape(-1, 2)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        box = dict(HPOS=int(x0), VPOS=int(y0), WIDTH=int(x1 - x0), HEIGHT=int(y1 - y0))

        block = _el(print_space, "TextBlock", ID=f"block_{i}", **box)
        line = _el(block, "TextLine", ID=f"line_{i}", **box)

        points_str = " ".join(f"{int(x)},{int(y)}" for x, y in pts)
        _el(_el(line, "Shape"), "Polygon", POINTS=points_str)

        _el(line, "String", ID=f"string_{i}", **box, CONTENT=text)

    etree.ElementTree(root).write(out_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


from PIL import ImageDraw, ImageFont
from fontTools.ttLib import TTFont as _TTFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CMAP_CACHE = {}


def _get_pil_font(size=22):
    candidates = [
        os.path.join(SCRIPT_DIR, "fonts", "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return path, ImageFont.truetype(path, size)
            except Exception:
                pass
    print("[!] Warning: no Devanagari-capable font found, falling back to PIL default.")
    return None, ImageFont.load_default()


def _get_cmap(font_path):
    if font_path not in _CMAP_CACHE:
        try:
            _CMAP_CACHE[font_path] = set(_TTFont(font_path, fontNumber=0).getBestCmap().keys())
        except Exception as e:
            print(f"[!] Could not read cmap from {font_path}: {e}")
            _CMAP_CACHE[font_path] = None
    return _CMAP_CACHE[font_path]


def _missing_glyphs(font_path, text):
    cmap = _get_cmap(font_path)
    if cmap is None:
        return set()
    return {ch for ch in text if not ch.isspace() and ord(ch) not in cmap}


def _fit_font_to_box(draw, text, box_w, box_h, max_size, min_size=6):
    for size in range(max_size, min_size - 1, -1):
        font_path, font = _get_pil_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= box_w and h <= box_h:
            return font_path, font
    return _get_pil_font(min_size)


def save_overlay_image(img, polys, texts, out_img_path, box_color=(0, 255, 0),
                        text_color=(255, 0, 0), max_font_size=30, draw_index=True):
    overlay_bgr = img.copy()
    for poly in polys:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay_bgr, [pts], isClosed=True, color=box_color, thickness=2)

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(pil_img)
    text_color_rgb = (text_color[2], text_color[1], text_color[0])

    for i, (poly, text) in enumerate(zip(polys, texts)):
        pts = np.asarray(poly).reshape(-1, 2)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        box_w, box_h = max(x_max - x_min, 1), max(y_max - y_min, 1)

        label = f"[{i}] {text}" if draw_index else text
        if not label.strip():
            continue

        font_path, font = _fit_font_to_box(draw, label, box_w, box_h, max_font_size)
        if font_path:
            missing = _missing_glyphs(font_path, text)
            if missing:
                codepoints = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in sorted(missing))
                print(f"[!] Line {i}: font has no glyph for: {codepoints}")

        draw.rectangle([x_min, y_min, x_max, y_max], fill=(255, 255, 255))
        draw.text((x_min, y_min), label, fill=text_color_rgb, font=font)

    pil_img.save(out_img_path)
    print(f"Saved overlay image -> {out_img_path}")

# ---------------------------------------------------------------------------
# Main pipeline: detect -> crop -> recognize -> save
# ---------------------------------------------------------------------------
def run_pipeline(image_path, results_root="results"):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(results_root, image_name)
    os.makedirs(out_dir, exist_ok=True)

    det_result = det_model.predict(image_path)

    for res in det_result:
        img = cv2.imread(res["input_path"])
        polys = sort_polys_reading_order(res["dt_polys"])

        # draw detected boxes for a quick visual sanity check
        boxes_img = img.copy()
        for poly in polys:
            pts = np.asarray(poly).astype(int).reshape(-1, 1, 2)
            cv2.polylines(boxes_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(out_dir, "boxes.jpg"), boxes_img)

        # recognize each line
        texts = []
        for i, poly in enumerate(polys):
            crop = get_rotate_crop_image(img, poly)
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            try:
                text = predict_text(pil_crop)
            except Exception as e:
                print(f"[!] Error on line {i}: {e}")
                text = ""
            texts.append(text)

        # save ALTO with recognized text embedded
        save_alto_xml(
            polys=polys,
            texts=texts,
            img_shape=img.shape,
            image_filename=os.path.basename(image_path),
            out_path=os.path.join(out_dir, "alto.xml"),
        )

        # save overlay image (boxes blanked + predicted text drawn in)
        save_overlay_image(
            img=img,
            polys=polys,
            texts=texts,
            out_img_path=os.path.join(out_dir, "overlay.jpg"),
        )

        # save plain text file too
        with open(os.path.join(out_dir, "predicted.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

    print(f"Done -> {out_dir}")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def run_pipeline_batch(images_dir, results_root="results"):
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"[!] No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images in {images_dir}")

    for idx, fname in enumerate(image_files, start=1):
        image_path = os.path.join(images_dir, fname)
        print(f"\n[{idx}/{len(image_files)}] Processing {fname} ...")
        try:
            run_pipeline(image_path, results_root=results_root)
        except Exception as e:
            print(f"[!] Failed on {fname}: {e}")


if __name__ == "__main__":
    run_pipeline_batch(
        images_dir="images/",
        results_root="results",
    )