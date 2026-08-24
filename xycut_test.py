from paddleocr import PPStructureV3

structure = PPStructureV3()
result = structure.predict("images/gorkhapatra_images/Ajitman Tamangs Article_Published in National Daily Gorkhapatra_Gaurab-sali Paramparaka_ Dhani Adivasi-Tamang.jpeg")

for res in result:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="outputs") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="outputs") ## Save the current image's result in Markdown format
    res.save_to_word(save_path="outputs") ## Save the current image's result in Word format
    res.save_to_img(save_path="outputs")


for res in result:
    print("Got a result, trying to save image...")
    try:
        res.save_to_img(save_path="outputs")
        print("SUCCESS: image saved")
    except Exception as e:
        print("ERROR while saving image:")
        print(e)