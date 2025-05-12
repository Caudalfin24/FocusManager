import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from emotion_net import EmotionNet

# 参数配置
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集

train_dir = 'data/FER2013/train'
test_dir = 'data/FER2013/test'

train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 加载模型

model = EmotionNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
# 早停变量
best_val_acc = 0.0
patience_counter = 0
save_path = "emotion_net_best.pth"

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}, Train Acc: {acc*100:.2f}%")
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = correct / len(test_loader.dataset)
    scheduler.step(val_acc)
    print(f"Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"-> Best model updated. Saved to {save_path}")
    else:
        patience_counter += 1
        print(f"-> No improvement. Patience: {patience_counter} / {PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print("-> Early stopping triggerd.")
            break

print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")