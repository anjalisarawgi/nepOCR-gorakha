from PIL import Image
from transformers import pipeline

pipe = pipeline(task="image-text-to-text", 
                # model="Qwen/Qwen2.5-VL-7B-Instruct",
                model = "Qwen/Qwen3.6-27B-FP8",
                device=0,
                # torch_dtype="auto", 
            )

image = Image.open("/home/ra65vat/ra65vat_LRZ_mount/gorkhapatra/images/K 120_00000196.jpeg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "This is a scanned document of handwritten text in Old Nepali. There also appears to be some side nodes on the left side and the top of the page. Please transcribe this image."},
        ],
    }
]

result = pipe(text=messages, max_new_tokens=2000, return_full_text=False)
print(result[0]["generated_text"])

# यात्रीहरूको आराम र सुविधा को विशेष ध्यान राख्ने नेपाल ट्रान्सपोर्ट समितिको डिल्कस बस
# * दिन दिन ठीक नियत समयमा काठमाडौं र अर्चलेखाँज जाने श्राउने गदेछन्।
# विधान 6 बजे सुन्दराराम।