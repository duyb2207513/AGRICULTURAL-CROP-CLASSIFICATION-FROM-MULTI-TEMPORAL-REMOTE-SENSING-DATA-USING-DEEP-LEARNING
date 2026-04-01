import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CropTimeSeriesDataset # File nay da co noi suy
from model import CropClassifier # File model da sua pooling
import os

# ================= CAU HINH =================
DATA_DIR = "dataset_bo_sung" # Doi ten folder du lieu cua ban o day
BATCH_SIZE = 16
LEARNING_RATE = 0.0001 # Tang nhe LR chut cho nhanh hoi tu
EPOCHS = 10            # Tang Epoch len 50 vi du lieu chuoi can hoc ky hon
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QUAN TRONG: Sua lai cho dung bai toan cua ban
INPUT_DIM = 9   # 9 kenh (NDVI, EVI, SWIR, v.v...)

def train():
    print(f"--- Dang chay tren thiet bi: {DEVICE} ---")
    
    # 1. Load Data
    train_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
    
    # Kiem tra xem co du lieu validation khong
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        print("CANH BAO: Khong tim thay tap val, se dung tap train de danh gia tam.")
        val_dataset = train_dataset

    # TU DONG CAP NHAT SO LOP
    num_classes = len(train_dataset.classes)
    print(f"Thong tin: So lop thuc te tim thay: {num_classes} -> {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Thong ke: Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    
    # 2. Khoi tao Model
    model = CropClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(DEVICE)
    
    # Ham mat mat va toi uu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Vong lap Training
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero grad
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Tinh toan
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / len(train_loader)
        
        # 4. Validation (Danh gia)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Luu model tot nhat
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Luu y: Da luu model tot nhat!")

    print(f"--- HOAN THANH! Best Val Acc: {best_acc:.2f}% ---")

if __name__ == "__main__":
    # Dam bao thu muc data ton tai
    if os.path.exists(DATA_DIR):
        train()
    else:
        print(f"LOI: Khong tim thay thu muc {DATA_DIR}!")