from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False, # dont autorotate the whole document 
    use_doc_unwarping=False,  # dont autocorrect the curvature
    # use_textline_orientation=False,
    engine="paddle", lang = "ne")

result = ocr.predict("images/K 120_00000196.jpeg")
for res in result:
    res.print()
    res.save_to_img("output")
    # res.save_to_json("output")


##############################################

# from paddleocr import PaddleOCRVL

# pipeline = PaddleOCRVL(engine="transformers")  

# output = pipeline.predict("images/CA__The Cambridge Manuscript_Page_008.png")

# for res in output:
#     res.print()
#     res.save_to_json(save_path="output")
#     res.save_to_img(save_path="output")