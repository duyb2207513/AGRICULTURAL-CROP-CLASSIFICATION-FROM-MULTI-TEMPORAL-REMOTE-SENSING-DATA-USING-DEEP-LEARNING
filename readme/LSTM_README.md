# 📋 HƯỚNG DẪN NHANH - LSTM với JSON History

## 🎯 Mục đích

File `train_lstm_updated.py` là phiên bản cập nhật của `train.py` với các tính năng mới:

✅ Lưu training history vào JSON (giống Transformer)
✅ Có learning rate scheduler
✅ Lưu checkpoint đầy đủ (model config + weights)
✅ UI đẹp hơn với emoji và formatting
✅ Tương thích hoàn toàn với compare_models.py

---

## 🚀 Cách sử dụng

### Bước 1: Thay thế file cũ (Khuyến nghị)

```bash
# Backup file cũ (nếu muốn)
cp train.py train_old.py

# Thay thế bằng version mới
cp train_lstm_updated.py train.py

# Hoặc chạy trực tiếp file mới
python train_lstm_updated.py
```

### Bước 2: Train lại LSTM

```bash
python train_lstm_updated.py
```

**Output:**
- `best_model.pth` - Model weights + config
- `training_history_lstm.json` - Training metrics

---

## 📊 So sánh với Transformer

### Bước 3: Visualize training curves

```bash
python visualize_training.py
```

**Tạo ra 3 biểu đồ:**
1. `training_curves_comparison.png` - So sánh LSTM vs Transformer
2. `lstm_training_analysis.png` - Chi tiết LSTM
3. `transformer_training_analysis.png` - Chi tiết Transformer

### Bước 4: So sánh metrics và confusion matrix

```bash
python compare_models.py
```

**Tạo ra:**
- Bar charts comparison
- Side-by-side confusion matrices
- Per-class accuracy comparison
- JSON report

---

## 📁 Files Structure

```
your_project/
├── dataset_bo_sung/          # Data
├── train_lstm_updated.py     # NEW: LSTM training with JSON
├── train_transformer.py      # Transformer training
├── compare_models.py         # Compare both models
├── visualize_training.py     # Plot training curves
├── best_model.pth            # LSTM weights
├── best_model_transformer.pth # Transformer weights
├── training_history_lstm.json      # NEW: LSTM history
└── training_history_transformer.json # Transformer history
```

---

## 🔍 JSON Format

File `training_history_lstm.json`:

```json
{
  "train_loss": [1.5, 1.2, 0.9, ...],
  "train_acc": [60.5, 75.2, 82.1, ...],
  "val_acc": [58.3, 72.1, 80.5, ...],
  "learning_rates": [0.0001, 0.0001, 0.00005, ...],
  "best_acc": 93.45,
  "best_epoch": 23,
  "total_params": 156234
}
```

---

## 📈 Visualization Examples

### 1. Quick plot in Python

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('training_history_lstm.json', 'r') as f:
    history = json.load(f)

# Plot validation accuracy
plt.plot(history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title(f"LSTM - Best: {history['best_acc']:.2f}%")
plt.grid()
plt.show()
```

### 2. Compare LSTM vs Transformer

```python
import json

lstm = json.load(open('training_history_lstm.json'))
transformer = json.load(open('training_history_transformer.json'))

print(f"LSTM Best: {lstm['best_acc']:.2f}%")
print(f"Transformer Best: {transformer['best_acc']:.2f}%")
print(f"Difference: {transformer['best_acc'] - lstm['best_acc']:.2f}%")
```

---

## 🎓 Cho Luận văn

### Table 1: Training Configuration

| Parameter | LSTM | Transformer |
|-----------|------|-------------|
| Batch Size | 16 | 16 |
| Learning Rate | 0.0001 | 0.0001 |
| Optimizer | Adam | AdamW |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Epochs | 50 | 50 |
| Total Params | ~156K | ~250K |

### Table 2: Results Comparison

| Metric | LSTM | Transformer | Improvement |
|--------|------|-------------|-------------|
| Best Val Acc | 93.00% | 96.88% | **+3.88%** |
| Train Time | ~25 min | ~20 min | Faster |
| Memory Usage | Lower | Higher | - |

### Figure 1: Training Curves

Dùng `training_curves_comparison.png`

Caption:
> "So sánh quá trình training giữa CNN-LSTM và CNN-Transformer. 
> Transformer đạt convergence nhanh hơn và accuracy cao hơn."

### Figure 2: Confusion Matrices

Dùng output từ `compare_models.py`

Caption:
> "Ma trận nhầm lẫn cho cả hai mô hình. Transformer giảm confusion 
> giữa các cặp class tương tự nhau."

---

## ⚙️ Advanced Options

### 1. Thay đổi số epochs

```python
# Trong train_lstm_updated.py
EPOCHS = 70  # Thay vì 50
```

### 2. Thay đổi learning rate schedule

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.5,      # Giảm 50% mỗi lần
    patience=5,      # Đợi 5 epochs
    min_lr=1e-7
)
```

### 3. Thêm early stopping

```python
# Thêm vào training loop
patience = 10
no_improve = 0

for epoch in range(EPOCHS):
    # ... training code ...
    
    if val_acc > best_acc:
        best_acc = val_acc
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

---

## 🔧 Troubleshooting

### Q: File JSON bị lỗi format?
```bash
# Kiểm tra JSON có hợp lệ không
python -m json.tool training_history_lstm.json
```

### Q: Muốn load lại để plot?
```python
import json

with open('training_history_lstm.json', 'r') as f:
    history = json.load(f)

# Giờ dùng history như dictionary bình thường
print(history['best_acc'])
```

### Q: So sánh 2 runs khác nhau?
```bash
# Đổi tên file cũ
mv training_history_lstm.json training_history_lstm_run1.json

# Train lại
python train_lstm_updated.py
# Tạo training_history_lstm.json mới

# So sánh
python
>>> import json
>>> run1 = json.load(open('training_history_lstm_run1.json'))
>>> run2 = json.load(open('training_history_lstm.json'))
>>> print(f"Run1: {run1['best_acc']:.2f}%")
>>> print(f"Run2: {run2['best_acc']:.2f}%")
```

---

## ✅ Checklist

Sau khi train xong:

- [ ] File `training_history_lstm.json` tồn tại
- [ ] File `best_model.pth` tồn tại
- [ ] Best accuracy >= 90%
- [ ] Chạy `visualize_training.py` thành công
- [ ] Chạy `compare_models.py` thành công
- [ ] Có đủ biểu đồ cho luận văn

---

**Chúc bạn thành công! 🎓🚀**