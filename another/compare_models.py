import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import os
import json
import pickle

from dataset_loader import CropTimeSeriesDataset
from model_ndvi_statistics import NDVIStatisticsClassifier
from model import CropClassifier  # LSTM
from model_transformer import CropClassifierTransformer, CropClassifierTransformerAdvanced

# ================= CẤU HÌNH =================
DATA_DIR = "dataset_bo_sung"
NDVI_STATS_MODEL_PATH = "best_model_ndvi_stats.pkl"
LSTM_MODEL_PATH = "best_model.pth"
TRANSFORMER_MODEL_PATH = "best_model_transformer.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 9

def convert_torch_to_numpy(dataloader):
    """
    Convert PyTorch DataLoader to NumPy arrays (for NDVI stats model)
    """
    all_data = []
    all_labels = []
    
    for inputs, labels in dataloader:
        all_data.append(inputs.numpy())
        all_labels.append(labels.numpy())
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

def load_ndvi_stats_model(model_path):
    """Load NDVI Statistics model (pickle file)"""
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def load_lstm_model(model_path):
    """Load LSTM model (PyTorch)"""
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Get dataset to determine num_classes
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
    
    num_classes = len(val_dataset.classes)
    
    # Try to get model config from checkpoint
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        config = checkpoint['model_config']
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config.get('num_layers', 2)
        print(f"  → Loaded LSTM config: hidden_dim={hidden_dim}, num_layers={num_layers}")
    else:
        # Infer from checkpoint weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Get hidden_dim from LSTM weight shape
        # lstm.weight_ih_l0 shape is (4*hidden_size, input_size)
        lstm_weight = state_dict.get('lstm.weight_ih_l0', None)
        if lstm_weight is not None:
            hidden_dim = lstm_weight.shape[0] // 4  # 512 // 4 = 128
            print(f"  → Inferred LSTM hidden_dim from weights: {hidden_dim}")
        else:
            hidden_dim = 128
            print(f"  → Using default hidden_dim: {hidden_dim}")
        
        # Check number of layers
        if 'lstm.weight_ih_l1' in state_dict:
            num_layers = 2
        else:
            num_layers = 1
        print(f"  → Detected num_layers: {num_layers}")
    
    # Create model with correct config
    model = CropClassifier(
        input_dim=INPUT_DIM, 
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, num_classes

def load_transformer_model(model_path):
    """Load Transformer model (PyTorch)"""
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
    
    num_classes = len(val_dataset.classes)
    
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model_variant = config.get('model_type', 'basic')
        
        if model_variant == 'basic':
            model = CropClassifierTransformer(
                input_dim=INPUT_DIM,
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 4),
                num_layers=config.get('num_layers', 2),
                num_classes=num_classes
            ).to(DEVICE)
        else:
            model = CropClassifierTransformerAdvanced(
                input_dim=INPUT_DIM,
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 4),
                num_layers=config.get('num_layers', 2),
                num_classes=num_classes
            ).to(DEVICE)
    else:
        model = CropClassifierTransformer(input_dim=INPUT_DIM, num_classes=num_classes).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, num_classes

def evaluate_pytorch_model(model, val_loader):
    """Evaluate PyTorch model (LSTM or Transformer)"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def evaluate_ndvi_stats_model(model, X_test, y_test):
    """Evaluate NDVI Statistics model (sklearn-based)"""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    return predictions, y_test, probabilities

def compare_all_models():
    """
    Main comparison function for all models
    Supports both PyTorch models and sklearn models
    """
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*10 + "COMPREHENSIVE MODEL COMPARISON" + " "*27 + "║")
    print("║" + " "*8 + "NDVI Stats vs LSTM vs Transformer" + " "*24 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Load dataset
    print("📂 Loading validation dataset...")
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='val', max_len=10)
    else:
        print("⚠️  Using train set for evaluation")
        val_dataset = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    classes_names = val_dataset.classes
    num_classes = len(classes_names)
    
    print(f"✓ Classes ({num_classes}): {classes_names}\n")
    
    # Convert to NumPy for NDVI stats model
    X_val, y_val = convert_torch_to_numpy(val_loader)
    print(f"✓ Converted to NumPy: {X_val.shape}\n")
    
    # Load models
    print("🔄 Loading models...")
    models = {}
    
    # 1. NDVI Statistics (pickle)
    ndvi_model = load_ndvi_stats_model(NDVI_STATS_MODEL_PATH)
    if ndvi_model is not None:
        models['NDVI Statistics'] = {'model': ndvi_model, 'type': 'ndvi_stats'}
        print("✓ NDVI Statistics model loaded (Traditional ML)")
    
    # 2. LSTM (PyTorch)
    lstm_model, _ = load_lstm_model(LSTM_MODEL_PATH)
    if lstm_model is not None:
        models['LSTM'] = {'model': lstm_model, 'type': 'pytorch'}
        print("✓ LSTM model loaded (Deep Learning)")
    
    # 3. Transformer (PyTorch)
    transformer_model, _ = load_transformer_model(TRANSFORMER_MODEL_PATH)
    if transformer_model is not None:
        models['Transformer'] = {'model': transformer_model, 'type': 'pytorch'}
        print("✓ Transformer model loaded (Deep Learning)")
    
    if len(models) == 0:
        print("\n❌ ERROR: No models found!")
        print("💡 Train models first:")
        print("   python train_ndvi_statistics.py")
        print("   python train_lstm_updated.py")
        print("   python train_transformer.py")
        return
    
    print(f"\n✓ Total models to compare: {len(models)}\n")
    
    # Evaluate all models
    results = {}
    
    for model_name, model_info in models.items():
        print(f"🔍 Evaluating {model_name}...")
        
        model = model_info['model']
        model_type = model_info['type']
        
        if model_type == 'ndvi_stats':
            # NDVI Stats: use NumPy arrays
            preds, labels, probs = evaluate_ndvi_stats_model(model, X_val, y_val)
        else:
            # PyTorch: use DataLoader
            # Recreate loader since we consumed it
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            preds, labels, probs = evaluate_pytorch_model(model, val_loader)
        
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        
        results[model_name] = {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
            'confusion_matrix': confusion_matrix(labels, preds)
        }
        
        print(f"  ✓ Accuracy: {acc*100:.2f}%")
        print(f"  ✓ F1-Macro: {f1_macro:.4f}")
        print(f"  ✓ F1-Weighted: {f1_weighted:.4f}\n")
    
    # ========== VISUALIZATION ==========
    
    # 1. Bar chart comparison
    print("📊 Creating comparison charts...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    f1_macros = [results[m]['f1_macro'] for m in model_names]
    f1_weighteds = [results[m]['f1_weighted'] for m in model_names]
    
    colors = ['#95E1D3', '#FF6B6B', '#4ECDC4']
    
    # Accuracy
    bars = axes[0].bar(model_names, accuracies, color=colors[:len(model_names)], alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    for i, (bar, v) in enumerate(zip(bars, accuracies)):
        axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # F1-Macro
    bars = axes[1].bar(model_names, f1_macros, color=colors[:len(model_names)], alpha=0.8)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score (Macro)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    for i, (bar, v) in enumerate(zip(bars, f1_macros)):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    # F1-Weighted
    bars = axes[2].bar(model_names, f1_weighteds, color=colors[:len(model_names)], alpha=0.8)
    axes[2].set_ylabel('F1-Score', fontsize=12)
    axes[2].set_title('F1-Score (Weighted)', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    for i, (bar, v) in enumerate(zip(bars, f1_weighteds)):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('final_comparison_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: final_comparison_metrics.png")
    plt.show()
    
    # 2. Confusion Matrices
    n_models = len(models)
    if n_models == 3:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    elif n_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        acc = result['accuracy']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes_names, yticklabels=classes_names,
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('True', fontsize=11)
        axes[idx].set_title(f'{model_name}\nAccuracy: {acc*100:.2f}%',
                          fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: final_confusion_matrices.png")
    plt.show()
    
    # 3. Per-class performance
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(num_classes)
    width = 0.25
    
    for idx, (model_name, result) in enumerate(results.items()):
        preds = result['predictions']
        labels = result['labels']
        
        per_class_acc = []
        for i in range(num_classes):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum(preds[class_mask] == i) / np.sum(class_mask)
                per_class_acc.append(class_acc * 100)
            else:
                per_class_acc.append(0)
        
        offset = width * (idx - (len(results)-1)/2)
        bars = ax.bar(x + offset, per_class_acc, width, label=model_name,
                     color=colors[idx], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Crop Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_per_class_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: final_per_class_comparison.png")
    plt.show()
    
    # ========== SUMMARY TABLE ==========
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<12} {'Improvement':<12}")
    print("-"*80)
    
    # Determine baseline
    baseline_name = 'NDVI Statistics' if 'NDVI Statistics' in results else list(results.keys())[0]
    baseline_acc = results[baseline_name]['accuracy']
    
    for model_name, result in results.items():
        acc = result['accuracy']
        improvement = "" if model_name == baseline_name else f"+{(acc - baseline_acc)*100:.2f}%"
        
        print(f"{model_name:<20} "
              f"{acc*100:>10.2f}% "
              f"{result['f1_macro']:>11.4f} "
              f"{result['f1_weighted']:>11.4f} "
              f"{improvement:>11}")
    
    # Winner
    print("\n" + "="*80)
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 BEST MODEL: {best_model[0]} ({best_model[1]['accuracy']*100:.2f}%)")
    
    # Analysis
    if len(results) >= 2:
        print("\n📊 KEY FINDINGS:")
        
        if 'NDVI Statistics' in results:
            baseline_acc = results['NDVI Statistics']['accuracy'] * 100
            
            if 'LSTM' in results:
                lstm_acc = results['LSTM']['accuracy'] * 100
                lstm_imp = lstm_acc - baseline_acc
                print(f"  • LSTM vs Traditional ML: +{lstm_imp:.2f}%")
            
            if 'Transformer' in results:
                trans_acc = results['Transformer']['accuracy'] * 100
                trans_imp = trans_acc - baseline_acc
                print(f"  • Transformer vs Traditional ML: +{trans_imp:.2f}%")
            
            if 'LSTM' in results and 'Transformer' in results:
                lstm_vs_trans = trans_acc - lstm_acc
                print(f"  • Transformer vs LSTM: +{lstm_vs_trans:.2f}%")
                
                print(f"\n  ✓ Temporal modeling (DL) improves: +{lstm_imp:.1f}% to +{trans_imp:.1f}%")
                print(f"  ✓ Attention mechanism adds: +{lstm_vs_trans:.1f}%")
    
    print("="*80)
    
    # Save comparison report
    comparison_report = {
        'models': {},
        'comparison_date': str(np.datetime64('now')),
        'baseline': baseline_name
    }
    
    for model_name, result in results.items():
        comparison_report['models'][model_name] = {
            'accuracy': float(result['accuracy']),
            'f1_macro': float(result['f1_macro']),
            'f1_weighted': float(result['f1_weighted']),
            'improvement_over_baseline': float(result['accuracy'] - baseline_acc) if model_name != baseline_name else 0.0
        }
    
    with open('final_comparison_report.json', 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print("\n✅ Comparison completed!")
    print("📄 Report saved to: final_comparison_report.json")
    print("\n📊 Generated visualizations:")
    print("  1. final_comparison_metrics.png")
    print("  2. final_confusion_matrices.png")
    print("  3. final_per_class_comparison.png")

if __name__ == "__main__":
    compare_all_models()