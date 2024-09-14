from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="lit",
)

prompt = "Describe this image in detail please."
image_url = "https://i.ytimg.com/vi/740pO-ljkZs/sddefault.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]
response = client.chat.completions.create(
    model="mistralai/Pixtral-12B-2409",
    messages=messages,
    max_tokens=512,
)

print("\033[92m {}\033[00m".format(response.choices[0].message.content))
