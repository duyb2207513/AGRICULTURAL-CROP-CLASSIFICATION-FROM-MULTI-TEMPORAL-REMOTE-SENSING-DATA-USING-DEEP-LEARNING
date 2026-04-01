import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

from dataset_loader import CropTimeSeriesDataset 
from model_transformer import CropClassifierTransformer, CropClassifierTransformerAdvanced

# ================= CẤU HÌNH =================
DATA_DIR = "dataset_bo_sung"
MODEL_PATH = "best_model_transformer.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 9

def evaluate():
    print(f"╔{'═'*60}╗")
    print(f"║ EVALUATING TRANSFORMER MODEL")
    print(f"║ Device: {DEVICE}")
    print(f"╚{'═'*60}╝\n")
    
    # 1. Load dữ liệu
    print("📂 Loading validation dataset...")
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        print("⚠️  WARNING: Không tìm thấy tập val, dùng tập train để test tạm.")
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    classes_names = val_dataset.classes
    num_classes = len(classes_names)
    print(f"✓ Số lớp: {num_classes}")
    print(f"✓ Danh sách: {classes_names}\n")
    
    # 2. Load Model
    print("🔄 Loading model...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Không tìm thấy file {MODEL_PATH}")
        print("💡 Hãy chạy train_transformer.py trước!")
        return
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Lấy thông tin config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model_type = config.get('model_type', 'basic')
        d_model = config.get('d_model', 128)
        nhead = config.get('nhead', 4)
        num_layers = config.get('num_layers', 2)
        
        print(f"✓ Model type: {model_type}")
        print(f"✓ Config: d_model={d_model}, nhead={nhead}, layers={num_layers}")
    else:
        # Fallback nếu không có config
        model_type = 'basic'
        d_model, nhead, num_layers = 128, 4, 2
        print("⚠️  No config found, using default settings")
    
    # Khởi tạo model
    if model_type == 'basic':
        model = CropClassifierTransformer(
            input_dim=INPUT_DIM,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(DEVICE)
    else:
        model = CropClassifierTransformerAdvanced(
            input_dim=INPUT_DIM,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(DEVICE)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"✓ Validation accuracy during training: {checkpoint.get('val_acc', 'N/A'):.2f}%\n")
    
    # 3. Evaluation
    print("🔍 Running evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Lưu probabilities để phân tích
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            
            # Forward
            outputs = model(inputs)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 4. Metrics
    if len(all_labels) == 0:
        print("❌ ERROR: Không có dữ liệu để đánh giá!")
        return

    acc = accuracy_score(all_labels, all_preds)
    
    print("="*70)
    print(f"🎯 OVERALL ACCURACY: {acc*100:.2f}%")
    print("="*70)
    
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-"*70)
    report = classification_report(
        all_labels, all_preds, 
        target_names=classes_names, 
        digits=4, 
        zero_division=0
    )
    print(report)
    
    # 5. Confusion Matrix
    print("\n📈 Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=classes_names, yticklabels=classes_names,
                cbar_kws={'label': 'Number of samples'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - Transformer Model\nAccuracy: {acc*100:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('confusion_matrix_transformer.png', dpi=300, bbox_inches='tight')
    print("✓ Saved to: confusion_matrix_transformer.png")
    plt.show()
    
    # 6. Per-class analysis
    print("\n" + "="*70)
    print("📋 PER-CLASS ANALYSIS:")
    print("="*70)
    
    for i, class_name in enumerate(classes_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.sum((np.array(all_preds)[class_mask] == i)) / np.sum(class_mask)
            class_samples = np.sum(class_mask)
            
            # Tính confidence trung bình cho class này
            class_probs = np.array(all_probs)[class_mask][:, i]
            avg_confidence = np.mean(class_probs)
            
            print(f"{class_name:15s} | "
                  f"Samples: {class_samples:4d} | "
                  f"Accuracy: {class_acc*100:6.2f}% | "
                  f"Avg Confidence: {avg_confidence:.4f}")
    
    # 7. Confusion pairs - Tìm cặp class hay bị nhầm lẫn
    print("\n" + "="*70)
    print("⚠️  MOST CONFUSED PAIRS:")
    print("="*70)
    
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], classes_names[i], classes_names[j]))
    
    confusion_pairs.sort(reverse=True)
    
    if confusion_pairs:
        for count, true_class, pred_class in confusion_pairs[:5]:
            print(f"  • {true_class:15s} → {pred_class:15s} : {count:3d} times")
    else:
        print("  🎉 No confusion! Perfect classification!")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    evaluate()
