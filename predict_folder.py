import os
import json
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import WeatherCNN

# ========== Config ==========
model_path = "best85_epoch16_weather_cnn.pth"  # change to your best model
class_names = ["cloudy", "foggy", "rainy", "shine", "sunrise"]

folder = "/Users/dingkwanmok/Desktop/test/split_dataset/train/cloudy"  # change to your folder path


device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

def load_model(model_path):
    model = WeatherCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def predict_folder(folder_path, model, output_json="predictions.json"):
    results = []
    image_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_files.append(os.path.join(root, file))

    for img_path in tqdm(image_files, desc="Predicting images", unit="img"):
        prediction = predict_image(img_path, model)
        results.append({
            "image_path": img_path,
            "prediction": prediction
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(results)} predictions to {output_json}")

if __name__ == "__main__":
    model = load_model(model_path)
    predict_folder(folder, model, output_json="predictions.json")