from PIL import Image
from transformers import pipeline

pipe = pipeline(task="image-text-to-text", 
                # model="Qwen/Qwen2.5-VL-7B-Instruct",
                model_kwargs={"cache_dir": "/home/ra65vat/ra65vat_LRZ_mount/gorkhapatra/hf_cache"},
                model = "Qwen/Qwen3-VL-8B-Instruct",
                # model = "Qwen/Qwen3.6-27B",
                # device=0,
                # torch_dtype="auto", 
            )

image = Image.open("/home/ra65vat/ra65vat_LRZ_mount/gorkhapatra/images/gorkhapatra_images/गोरखापत्रको पहिलो अंक First Issue of Gorkhapatra Daily_0004.jpeg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "This ."},
        ],
    }
]

result = pipe(text=messages, max_new_tokens=10000, return_full_text=False)
print(result[0]["generated_text"])



# --- save prompt + generated text to qwen/ dir ---
out_dir = "qwen"
os.makedirs(out_dir, exist_ok=True)

image_name = os.path.splitext(os.path.basename(image_path))[0]
out_path = os.path.join(out_dir, f"{image_name}.txt")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"PROMPT:\n{prompt_text}\n\n")
    f.write(f"GENERATED TEXT:\n{generated_text}\n")

print(f"saved to::: {out_path}")

# यात्रीहरूको आराम र सुविधा को विशेष ध्यान राख्ने नेपाल ट्रान्सपोर्ट समितिको डिल्कस बस
# * दिन दिन ठीक नियत समयमा काठमाडौं र अर्चलेखाँज जाने श्राउने गदेछन्।
# विधान 6 बजे सुन्दराराम।