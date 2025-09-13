import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ============== 配置 ==============
data_dir = "/Users/dingkwanmok/Desktop/test/split_dataset"   # 你的 train/val 路径
batch_size = 32
num_epochs = 30
learning_rate = 0.001

device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print("Using device:", device)

# ============== 数据增强 & 加载 ==============
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

classes = train_dataset.classes
print("Classes:", classes)

# ============== 自定义 CNN 模型 ==============
class WeatherCNN(nn.Module):
    def __init__(self, num_classes):
        super(WeatherCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 32, 64, 64]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [B, 64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [B, 128, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 128, 16, 16]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 40 * 40, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = WeatherCNN(num_classes=len(classes)).to(device)

# ============== 损失函数 & 优化器 ==============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_acc = 0.0

# ============== 训练循环 ==============
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs-1} [Train]", leave=False)

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())  # 动态显示当前 batch 的 loss

    # 验证
    model.eval()
    correct, total = 0, 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs-1} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch [{epoch}/{num_epochs-1}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    # ============== 保存模型 ==============

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_name = f"best{int(val_acc)}_epoch{epoch}_weather_cnn.pth"
        torch.save(model.state_dict(), model_name)
        print(f"Saved new best model: {model_name} with Val Acc: {best_val_acc:.2f}%")