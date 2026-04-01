import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import CropTimeSeriesDataset
from model_ndvi_statistics import NDVIStatisticsClassifier
import os
import json
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ================= CẤU HÌNH =================
DATA_DIR = "dataset_bo_sung"
BATCH_SIZE = 16  # Không quan trọng lắm vì không train neural network
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chọn classifier: 'random_forest' hoặc 'svm'
CLASSIFIER_TYPE = 'random_forest'  # Random Forest thường tốt hơn

# Dùng tất cả bands hay chỉ NDVI?
USE_ALL_BANDS = False  # False = chỉ dùng NDVI (band thứ 7)

def convert_torch_to_numpy(dataloader):
    """
    Convert PyTorch DataLoader to NumPy arrays
    """
    all_data = []
    all_labels = []
    
    for inputs, labels in dataloader:
        # Convert to numpy
        all_data.append(inputs.numpy())
        all_labels.append(labels.numpy())
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

def train():
    print(f"╔{'═'*60}╗")
    print(f"║ TRAINING NDVI TEMPORAL STATISTICS BASELINE")
    print(f"║ Method: Traditional Machine Learning (NOT Deep Learning)")
    print(f"║ Classifier: {CLASSIFIER_TYPE.upper()}")
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
    class_names = train_dataset.classes
    
    print(f"✓ Số lớp: {num_classes} -> {class_names}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")
    
    # 2. Convert to NumPy (vì model này không dùng PyTorch)
    print("🔄 Converting data to NumPy format...")
    X_train, y_train = convert_torch_to_numpy(train_loader)
    X_val, y_val = convert_torch_to_numpy(val_loader)
    
    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ X_val shape: {X_val.shape}\n")
    
    # 3. Initialize Model
    print(f"🏗️  Initializing {CLASSIFIER_TYPE.upper()} model...")
    print(f"✓ Feature extraction: NDVI temporal statistics")
    print(f"✓ Using {'all bands' if USE_ALL_BANDS else 'NDVI only (band 7)'}\n")
    
    model = NDVIStatisticsClassifier(
        classifier_type=CLASSIFIER_TYPE,
        use_all_bands=USE_ALL_BANDS
    )
    
    # 4. Train
    print("="*70)
    print("🚀 Starting training...")
    print("="*70)
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate on validation set
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    
    val_accuracy, val_preds, val_cm = model.evaluate(X_val, y_val, class_names)
    
    # Calculate F1 scores
    f1_macro = f1_score(y_val, val_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_val, val_preds, average='weighted', zero_division=0)
    
    print(f"\n✅ FINAL RESULTS:")
    print(f"  • Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"  • F1-Macro: {f1_macro:.4f}")
    print(f"  • F1-Weighted: {f1_weighted:.4f}")
    
    # 6. Save model
    print("\n💾 Saving model...")
    with open('best_model_ndvi_stats.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Saved to: best_model_ndvi_stats.pkl")
    
    # 7. Save training history (để so sánh với deep learning models)
    history = {
        'val_acc': val_accuracy * 100,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classifier_type': CLASSIFIER_TYPE,
        'use_all_bands': USE_ALL_BANDS,
        'n_features': len(model.feature_extractor.get_feature_names()),
        'class_names': class_names
    }
    
    with open('training_history_ndvi_stats.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✓ Saved history to: training_history_ndvi_stats.json")
    
    # 8. Plot confusion matrix
    print("\n📈 Creating confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - NDVI Statistics Baseline\n'
             f'{CLASSIFIER_TYPE.upper()} - Accuracy: {val_accuracy*100:.2f}%',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_ndvi_stats.png', dpi=300, bbox_inches='tight')
    print("✓ Saved to: confusion_matrix_ndvi_stats.png")
    plt.show()
    
    # 9. Feature importance plot (chỉ cho Random Forest)
    if CLASSIFIER_TYPE == 'random_forest':
        print("\n📊 Creating feature importance plot...")
        
        importances = model.classifier.feature_importances_
        feature_names = model.feature_extractor.get_feature_names()
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[indices], alpha=0.8)
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title('Feature Importance - NDVI Temporal Statistics', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance_ndvi_stats.png', dpi=300, bbox_inches='tight')
        print("✓ Saved to: feature_importance_ndvi_stats.png")
        plt.show()
    
    return history

def compare_with_deep_learning():
    """
    So sánh với LSTM và Transformer
    """
    print("\n" + "="*70)
    print("📊 COMPARING WITH DEEP LEARNING MODELS")
    print("="*70)
    
    results = {}
    
    # Load NDVI stats results
    if os.path.exists('training_history_ndvi_stats.json'):
        with open('training_history_ndvi_stats.json', 'r') as f:
            results['NDVI Stats'] = json.load(f)
        print(f"✓ NDVI Statistics: {results['NDVI Stats']['val_acc']:.2f}%")
    
    # Load LSTM
    if os.path.exists('training_history_lstm.json'):
        with open('training_history_lstm.json', 'r') as f:
            lstm_hist = json.load(f)
            results['LSTM'] = {'val_acc': lstm_hist['best_acc']}
        print(f"✓ LSTM: {results['LSTM']['val_acc']:.2f}%")
    
    # Load Transformer
    if os.path.exists('training_history_transformer.json'):
        with open('training_history_transformer.json', 'r') as f:
            trans_hist = json.load(f)
            results['Transformer'] = {'val_acc': trans_hist['best_acc']}
        print(f"✓ Transformer: {results['Transformer']['val_acc']:.2f}%")
    
    if len(results) < 2:
        print("\n⚠️  Need at least 2 models for comparison")
        return
    
    # Summary table
    print("\n" + "="*70)
    print("📋 COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Type':<20} {'Improvement':<15}")
    print("-"*70)
    
    baseline_acc = results.get('NDVI Stats', {}).get('val_acc', 0)
    
    # NDVI Stats
    if 'NDVI Stats' in results:
        print(f"{'NDVI Statistics':<20} {baseline_acc:>10.2f}% {'Traditional ML':<20} {'(Baseline)':<15}")
    
    # LSTM
    if 'LSTM' in results:
        lstm_acc = results['LSTM']['val_acc']
        improvement = f"+{lstm_acc - baseline_acc:.2f}%"
        print(f"{'LSTM':<20} {lstm_acc:>10.2f}% {'Deep Learning':<20} {improvement:<15}")
    
    # Transformer
    if 'Transformer' in results:
        trans_acc = results['Transformer']['val_acc']
        improvement = f"+{trans_acc - baseline_acc:.2f}%"
        print(f"{'Transformer':<20} {trans_acc:>10.2f}% {'Deep Learning':<20} {improvement:<15}")
    
    print("="*70)
    
    # Analysis
    print("\n🎯 ANALYSIS:")
    if 'LSTM' in results and 'Transformer' in results:
        lstm_imp = results['LSTM']['val_acc'] - baseline_acc
        trans_imp = results['Transformer']['val_acc'] - baseline_acc
        
        print(f"\n1. Deep Learning vs Traditional ML:")
        print(f"   • LSTM improves by: +{lstm_imp:.2f}%")
        print(f"   • Transformer improves by: +{trans_imp:.2f}%")
        
        if lstm_imp > 5:
            print(f"\n2. ✓ Deep learning SIGNIFICANTLY better!")
            print(f"   → Proves temporal modeling is important")
        else:
            print(f"\n2. ⚠️  Improvement is modest")
            print(f"   → May need more data or better features")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Không tìm thấy thư mục {DATA_DIR}!")
        exit(1)
    
    try:
        # Training
        history = train()
        
        # Compare with deep learning
        compare_with_deep_learning()
        
        print("\n" + "="*70)
        print("🎓 HƯỚNG DẪN TIẾP THEO:")
        print("="*70)
        print("1. Model đã được lưu: best_model_ndvi_stats.pkl")
        print("2. Chạy compare_all_models.py để so sánh 3 models:")
        print("   • NDVI Statistics (Baseline)")
        print("   • LSTM")
        print("   • Transformer")
        print("\n3. Kỳ vọng kết quả:")
        print("   • NDVI Stats: 70-85% (Traditional ML)")
        print("   • LSTM: 90-95% (+10-15% improvement)")
        print("   • Transformer: 94-97% (+12-20% improvement)")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()