import cv2
import torch
from PIL import Image
import mobileclip
import os
from collections import deque
import openai
import base64
import csv
from io import BytesIO


openai.api_key = 'sk-xxx'  # Ensure your API key is set

labels = ["book", "something else", "open palm"]

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained="checkpoints/mobileclip_s0.pt"
)
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

text = tokenizer(labels)

BUFFER_MAX_LEN = 50
TO_QUALIFY = 0.2
FRAME_PERCENT = BUFFER_MAX_LEN * TO_QUALIFY
label_buffer = deque(maxlen=BUFFER_MAX_LEN)
recorded_book_vectors = []
book_count = 0
BREAK_PROMPT = "open palm"
BREAK_PROMPT_BUFFER_SIZE = 10

if not os.path.exists("books"):
    os.makedirs("books")

device = torch.device("cpu")
model.to(device)

def embedding_has_not_been_recorded(embedding):
    for recorded_book in recorded_book_vectors:
        if 100.0 * embedding @ recorded_book.T > 50:
            return False
    return True

def compress_image(image_path, quality=30, max_size=(800, 800)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_image_data(image_path):
    compressed_image = compress_image(image_path)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the book in this image. Return only the author and title in this format: Author: [author name]\nTitle: [book title]"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{compressed_image}"}}
                ],
            },
        ],
        max_tokens=100
    )

    result = response.choices[0].message["content"]
    author = result.split("\n")[0].split(":")[1].strip()
    title = result.split("\n")[1].split(":")[1].strip()

    return {"author": author, "title": title}


def add_to_csv(data, filename="results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["author", "title"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Failed to open webcam")
    exit()

with torch.no_grad():
    text_features = model.encode_text(text).to(device)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame")
            break

        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)

        buffer_count = label_buffer.count("book")

        if buffer_count > FRAME_PERCENT and embedding_has_not_been_recorded(image_features):
            book_count += 1
            image_path = f"books/book_{book_count}.jpg"
            cv2.imwrite(image_path, frame)
            label_buffer = []
            recorded_book_vectors.append(image_features)
            print(f"Recorded book {book_count}")
    
            try:
                book_data = get_image_data(image_path)
                add_to_csv(book_data)
                print(f"Added book {book_count} to CSV: {book_data}")
            except Exception as e:
                print(f"Error processing book {book_count}: {str(e)}")
                add_to_csv({"author": "Error", "title": f"Processing failed for book {book_count}"})

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        top_result = labels[torch.argmax(text_probs)]
        label_buffer.append(top_result)

        if label_buffer.count(BREAK_PROMPT) > BREAK_PROMPT_BUFFER_SIZE:
            break

        frame = cv2.putText(
            frame,
            top_result,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Books recorded: {book_count}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()
