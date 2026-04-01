import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

# Import lai model va dataset
# Luu y: Dam bao file dataset_loader.py va model.py nam cung thu muc
from dataset_loader import CropTimeSeriesDataset 
from model import CropClassifier 

# ================= CAU HINH =================
DATA_DIR = "dataset_bo_sung"  # Duong dan thu muc du lieu thuc te
MODEL_PATH = "best_model.pth" # File model da luu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QUAN TRONG: Phai khop voi luc train
INPUT_DIM = 9  # (NDVI, EVI, SWIR, v.v...)

def evaluate():
    print(f"--- Dang danh gia model tren thiet bi: {DEVICE} ---")
    
    # 1. Load du lieu Validation truoc de lay danh sach Lop (Classes)
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        print("CANH BAO: Khong tim thay tap val, dung tap train de test tam.")
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Lay danh sach lop tu dong
    classes_names = val_dataset.classes
    num_classes = len(classes_names)
    print(f"Danh sach lop ({num_classes}): {classes_names}")
    
    # 2. Load Model
    # Khoi tao khung model voi kich thuoc dung (9 input, N classes)
    model = CropClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        # map_location giup load model train bang GPU len may CPU khong bi loi
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"OK: Da load trong so tu {MODEL_PATH}")
    else:
        print(f"LOI: Chua tim thay file {MODEL_PATH}. Hay train model truoc!")
        return

    # 3. Chay du doan
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Dang chay du doan...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            
            # Forward
            outputs = model(inputs)
            
            # Lay nhan co xac suat cao nhat
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Tinh toan cac chi so
    if len(all_labels) == 0:
        print("LOI: Khong co du lieu de danh gia!")
        return

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nKET QUA: Do chinh xac tong the (Accuracy): {acc*100:.2f}%")
    
    print("\nBAO CAO CHI TIET (Classification Report):")
    # target_names giup in ra ten cay (Lua, Mia...) thay vi so (0, 1...)
    print(classification_report(all_labels, all_preds, target_names=classes_names, digits=4, zero_division=0))

    # 5. Ve Ma tran nham lan (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_names, yticklabels=classes_names)
    plt.xlabel('May du doan (Predicted)')
    plt.ylabel('Thuc te (Actual)')
    plt.title(f'Ma tran nham lan (Accuracy: {acc*100:.1f}%)')
    plt.show()

if __name__ == "__main__":
    evaluate()