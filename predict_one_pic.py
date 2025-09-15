import torch
from torchvision import transforms
from PIL import Image
from model import WeatherCNN

# ========== Config ==========
model_path = "best85_epoch16_weather_cnn.pth"  # change to your model file name
class_names = ["cloudy", "foggy", "rainy", "shine", "sunrise"]  # class order must match training
test_image = "/Users/dingkwanmok/Desktop/test/split_dataset/val/foggy/foggy3.jpeg"  # change to your test image path

device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# ========== Preprocessing ==========
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# ========== Load Model ==========
def load_model(model_path):
    model = WeatherCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ========== Predict Function ==========
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    return class_names[class_idx]

if __name__ == "__main__":
    model = load_model(model_path)
    result = predict(test_image, model)
    print(f"Prediction for {test_image}: {result}")