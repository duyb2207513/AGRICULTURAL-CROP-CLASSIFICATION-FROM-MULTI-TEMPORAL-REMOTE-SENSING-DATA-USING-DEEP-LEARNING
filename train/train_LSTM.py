import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CropTimeSeriesDataset
from model import CropClassifier
import os
import json

# ================= CẤU HÌNH =================
DATA_DIR = "dataset_bo_sung"
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 50  # Đã tăng lên 50 như Transformer để công bằng
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QUAN TRỌNG: Sửa lại cho đúng bài toán của bạn
INPUT_DIM = 9   # 9 kênh (NDVI, EVI, SWIR, v.v...)

def train():
    print(f"╔{'═'*60}╗")
    print(f"║ TRAINING CNN + LSTM MODEL")
    print(f"║ Device: {DEVICE}")
    print(f"╚{'═'*60}╝\n")
    
    # 1. Load Data
    print("📂 Loading datasets...")
    train_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
    
    # Kiểm tra xem có dữ liệu validation không
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        print("⚠️  WARNING: Không tìm thấy tập val, sẽ dùng tập train để đánh giá tạm.")
        val_dataset = train_dataset

    # TỰ ĐỘNG CẬP NHẬT SỐ LỚP
    num_classes = len(train_dataset.classes)
    print(f"✓ Số lớp: {num_classes} -> {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")
    
    # 2. Khởi tạo Model
    print("🏗️  Initializing LSTM model...")
    model = CropClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(DEVICE)
    
    # Đếm số parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Architecture: CNN → Bi-LSTM → Classifier\n")
    
    # Hàm mất mát và tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler (optional, giống Transformer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # 3. Vòng lặp Training
    print("🚀 Starting training...")
    print("="*70)
    
    best_acc = 0.0
    best_epoch = 0
    
    # Lưu lại training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    for epoch in range(EPOCHS):
        # ============ TRAINING ============
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
            
            # Tính toán
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / len(train_loader)
        
        # ============ VALIDATION ============
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
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        lr_reduced = current_lr < old_lr
        
        # Lưu history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:6.2f}% | "
              f"Val Acc: {val_acc:6.2f}% | "
              f"LR: {current_lr:.6f}", end="")
        
        # Show LR reduction
        if lr_reduced:
            print(f" 📉 LR reduced: {old_lr:.6f} → {current_lr:.6f}", end="")
        
        # Lưu model tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            
            # Lưu checkpoint đầy đủ (giống Transformer)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_config': {
                    'input_dim': INPUT_DIM,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'num_classes': num_classes,
                    'model_type': 'lstm'
                }
            }, "best_model.pth")
            print(" ← 🌟 BEST!")
        else:
            print()

    print("="*70)
    print(f"\n✅ HOÀN THÀNH!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # Lưu training history
    history['best_acc'] = best_acc
    history['best_epoch'] = best_epoch
    history['total_params'] = total_params
    
    with open('training_history_lstm.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved to: training_history_lstm.json")
    print(f"💾 Best model saved to: best_model.pth\n")
    
    return history

if __name__ == "__main__":
    # Đảm bảo thư mục data tồn tại
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Không tìm thấy thư mục {DATA_DIR}!")
        print("💡 Hãy đảm bảo dữ liệu đã được chuẩn bị đúng cách.")
        exit(1)
    
    try:
        # Training
        history = train()
        
        print("\n" + "="*70)
        print("🎓 HƯỚNG DẪN TIẾP THEO:")
        print("="*70)
        print("1. Chạy train_transformer.py để train Transformer")
        print("2. Chạy compare_models.py để so sánh LSTM vs Transformer")
        print("3. File training_history_lstm.json chứa toàn bộ metrics")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("💾 Model đã được lưu ở epoch cuối cùng.")
        
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {str(e)}")
        raise