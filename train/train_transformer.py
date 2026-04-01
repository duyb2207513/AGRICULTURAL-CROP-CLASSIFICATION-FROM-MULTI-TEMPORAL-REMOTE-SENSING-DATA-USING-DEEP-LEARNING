import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CropTimeSeriesDataset
from model_transformer import CropClassifierTransformer, CropClassifierTransformerAdvanced
import os
import json
from datetime import datetime

# ================= CẤU HÌNH =================
DATA_DIR = "dataset_bo_sung"
BATCH_SIZE = 16
LEARNING_RATE = 0.0001  # Transformer thường cần LR thấp hơn
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cấu hình Transformer
INPUT_DIM = 9
D_MODEL = 128        # Dimension của Transformer (phải chia hết cho nhead)
NHEAD = 4            # Số attention heads (4 hoặc 8 là phổ biến)
NUM_LAYERS = 3       # Số lớp Transformer (2-4 là đủ cho dữ liệu nhỏ/trung bình)
DROPOUT = 0.2

# Chọn loại model: 'basic' hoặc 'advanced'
MODEL_TYPE = 'basic'  # Đổi thành 'advanced' để dùng CLS token

def train():
    print(f"╔{'═'*60}╗")
    print(f"║ TRAINING CNN + TRANSFORMER MODEL")
    print(f"║ Device: {DEVICE}")
    print(f"║ Model Type: {MODEL_TYPE.upper()}")
    print(f"╚{'═'*60}╝\n")
    
    # 1. Load Data
    print("📂 Loading datasets...")
    train_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
    
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        print("⚠️  WARNING: Không tìm thấy tập val, sử dụng tập train để đánh giá tạm.")
        val_dataset = train_dataset

    num_classes = len(train_dataset.classes)
    print(f"✓ Số lớp: {num_classes} -> {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")
    
    # 2. Khởi tạo Model
    print("🏗️  Initializing Transformer model...")
    
    if MODEL_TYPE == 'basic':
        model = CropClassifierTransformer(
            input_dim=INPUT_DIM,
            cnn_hidden=128,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            dropout=DROPOUT
        ).to(DEVICE)
    else:  # advanced
        model = CropClassifierTransformerAdvanced(
            input_dim=INPUT_DIM,
            cnn_hidden=128,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            dropout=DROPOUT
        ).to(DEVICE)
    
    # Đếm số parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Architecture: CNN → Transformer ({NUM_LAYERS} layers, {NHEAD} heads)\n")
    
    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning Rate Scheduler - FIXED: removed verbose parameter
    # ReduceLROnPlateau sẽ tự động in ra khi giảm learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # maximize validation accuracy
        factor=0.5,      # giảm LR xuống 50%
        patience=5,      # đợi 5 epochs không cải thiện
        min_lr=1e-7      # LR tối thiểu
    )
    
    # 4. Training Loop
    print("🚀 Starting training...")
    print("="*70)
    
    best_acc = 0.0
    best_epoch = 0
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
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping - quan trọng cho Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
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
        
        # Update learning rate based on validation accuracy
        # Scheduler sẽ tự động print message khi giảm LR
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if LR was reduced
        lr_reduced = current_lr < old_lr
        
        # Save history
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
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_config': {
                    'input_dim': INPUT_DIM,
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'num_classes': num_classes,
                    'model_type': MODEL_TYPE
                }
            }, "best_model_transformer.pth")
            print(" ← 🌟 BEST!")
        else:
            print()
    
    print("="*70)
    print(f"\n✅ HOÀN THÀNH!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # Save training history
    history['best_acc'] = best_acc
    history['best_epoch'] = best_epoch
    history['total_params'] = total_params
    
    with open('training_history_transformer.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved to: training_history_transformer.json")
    print(f"💾 Best model saved to: best_model_transformer.pth\n")
    
    return history

def compare_with_lstm():
    """
    So sánh kết quả với mô hình LSTM nếu có
    """
    print("\n" + "="*70)
    print("📊 COMPARING WITH LSTM MODEL")
    print("="*70)
    
    # Load Transformer results
    if os.path.exists('training_history_transformer.json'):
        with open('training_history_transformer.json', 'r') as f:
            transformer_hist = json.load(f)
        print(f"✓ Transformer Best Val Acc: {transformer_hist['best_acc']:.2f}%")
    
    # Load LSTM results nếu có
    if os.path.exists('training_history_lstm.json'):
        with open('training_history_lstm.json', 'r') as f:
            lstm_hist = json.load(f)
        print(f"✓ LSTM Best Val Acc: {lstm_hist.get('best_acc', 'N/A'):.2f}%")
        
        # So sánh
        if 'best_acc' in lstm_hist:
            diff = transformer_hist['best_acc'] - lstm_hist['best_acc']
            if diff > 0:
                print(f"\n🎯 Transformer tốt hơn LSTM: +{diff:.2f}%")
            else:
                print(f"\n📈 LSTM vẫn tốt hơn: {abs(diff):.2f}%")
    else:
        print("⚠️  Chưa có kết quả LSTM để so sánh")
        print("💡 Chạy train.py trước để có baseline LSTM")

if __name__ == "__main__":
    # Kiểm tra dữ liệu
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Không tìm thấy thư mục {DATA_DIR}!")
        print("💡 Hãy đảm bảo dữ liệu đã được chuẩn bị đúng cách.")
        exit(1)
    
    try:
        # Training
        history = train()
        
        # So sánh với LSTM
        compare_with_lstm()
        
        print("\n" + "="*70)
        print("🎓 HƯỚNG DẪN TIẾP THEO:")
        print("="*70)
        print("1. Chạy evaluate_transformer.py để đánh giá chi tiết")
        print("2. Chạy compare_models.py để so sánh trực quan LSTM vs Transformer")
        print("3. Thử điều chỉnh hyperparameters:")
        print("   - Tăng NUM_LAYERS (3→4) nếu dữ liệu nhiều")
        print("   - Tăng NHEAD (4→8) để model học phức tạp hơn")
        print("   - Tăng D_MODEL (128→256) nếu GPU đủ mạnh")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("💾 Model đã được lưu ở epoch cuối cùng.")
        
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Kiểm tra CUDA/GPU availability")
        print("2. Giảm BATCH_SIZE nếu out of memory")
        print("3. Kiểm tra data loader có hoạt động không")
        print("4. Đảm bảo D_MODEL chia hết cho NHEAD")
        raise