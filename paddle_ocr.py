from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="paddle",
    lang = "hi"
)
result = ocr.predict("images/Gorkhapatra_masthead_9Jan33.jpg")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")