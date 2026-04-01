# 🌾 Phân loại Cây trồng Nông nghiệp - CNN+LSTM vs CNN+Transformer

## 📋 Tổng quan

Project này so sánh 2 kiến trúc học sâu cho bài toán phân loại cây trồng từ ảnh vệ tinh đa thời gian:

1. **CNN + LSTM** (Baseline đã đạt 93% accuracy)
2. **CNN + Transformer** (Model mới với Multi-Head Attention)

## 🏗️ Kiến trúc Model

### 1. CNN + LSTM (Baseline)

```
Input (Batch, 10, 9)
    ↓
CNN Feature Extractor
    • Conv1d (9→64→128 channels)
    • BatchNorm + ReLU + Dropout
    ↓
Bi-LSTM (128→128 hidden)
    • 2 layers, bidirectional
    • Học temporal patterns
    ↓
Global Average Pooling
    ↓
Classifier (256→64→5)
    ↓
Output (Batch, 5)
```

**Ưu điểm:**
- Hiệu quả với chuỗi thời gian ngắn
- Ít tham số hơn, train nhanh
- Đã đạt 93% accuracy

**Nhược điểm:**
- Khó học long-range dependencies
- Phải xử lý tuần tự (không parallel)

---

### 2. CNN + Transformer (Mới)

```
Input (Batch, 10, 9)
    ↓
CNN Feature Extractor
    • Conv1d (9→64→128 channels)
    • BatchNorm + GELU + Dropout
    ↓
Feature Projection (128→d_model)
    ↓
Positional Encoding
    • Thêm thông tin vị trí thời gian
    ↓
Transformer Encoder
    • Multi-Head Attention (4 heads)
    • Feed-Forward Networks
    • 3 layers
    ↓
Global Average Pooling
    ↓
Classifier (128→64→5)
    ↓
Output (Batch, 5)
```

**Ưu điểm:**
- **Multi-Head Attention**: Học được mối quan hệ giữa các time steps
- **Parallel Processing**: Train nhanh hơn LSTM
- **Long-range Dependencies**: Tốt hơn cho chuỗi dài
- **Interpretability**: Có thể visualize attention weights

**Nhược điểm:**
- Nhiều tham số hơn
- Cần dữ liệu nhiều hơn để tránh overfit

---

## 📊 So sánh Chi tiết

| Đặc điểm | LSTM | Transformer |
|----------|------|-------------|
| **Parameters** | ~150K | ~250K |
| **Training Speed** | Chậm (sequential) | Nhanh (parallel) |
| **Memory** | Ít hơn | Nhiều hơn |
| **Long-range** | Hạn chế | Tốt |
| **Interpretability** | Khó | Dễ (attention weights) |

---

## 🚀 Hướng dẫn Sử dụng

### Bước 1: Chuẩn bị Dữ liệu

Đảm bảo thư mục dữ liệu có cấu trúc:

```
dataset_bo_sung/
├── train/
│   ├── lúa/
│   │   ├── polygon_001/
│   │   │   ├── image_t1.tif
│   │   │   ├── image_t2.tif
│   │   │   └── ...
│   │   └── polygon_002/
│   ├── mía/
│   └── ...
└── val/
    └── (tương tự train)
```

### Bước 2: Train LSTM (Baseline)

```bash
python train.py
```

Kết quả:
- Model: `best_model.pth`
- Accuracy: ~93%

### Bước 3: Train Transformer

```bash
python train_transformer.py
```

Tùy chỉnh hyperparameters trong file:

```python
D_MODEL = 128        # Dimension (128, 256, 512)
NHEAD = 4            # Attention heads (4, 8)
NUM_LAYERS = 3       # Transformer layers (2, 3, 4)
LEARNING_RATE = 0.0001
EPOCHS = 50
```

**Lưu ý:** Transformer cần train nhiều epochs hơn LSTM!

### Bước 4: Đánh giá Transformer

```bash
python evaluate_transformer.py
```

Output:
- Classification report
- Confusion matrix
- Per-class analysis

### Bước 5: So sánh Models

```bash
python compare_models.py
```

Tạo ra:
1. `model_comparison_metrics.png` - Bar charts
2. `model_comparison_confusion_matrices.png` - Side-by-side confusion matrices
3. `model_comparison_per_class.png` - Per-class accuracy
4. `model_comparison_report.json` - Detailed report

---

## 🎯 Hyperparameter Tuning

### Cho Transformer

1. **D_MODEL** (Transformer dimension):
   - 128: Nhỏ gọn, ít overfit
   - 256: Cân bằng (khuyến nghị)
   - 512: Mạnh nhưng cần data nhiều

2. **NHEAD** (Số attention heads):
   - 4: Standard, ổn định
   - 8: Tốt hơn nếu d_model=256
   - Quy tắc: d_model phải chia hết cho nhead

3. **NUM_LAYERS**:
   - 2: Nhanh, ít overfit
   - 3: Cân bằng (khuyến nghị)
   - 4+: Cần dataset lớn

4. **LEARNING_RATE**:
   - Transformer: 0.0001 - 0.0003
   - Dùng ReduceLROnPlateau scheduler

5. **DROPOUT**:
   - 0.1-0.2: Standard
   - 0.3-0.4: Nếu overfit

---

## 📈 Kỳ vọng Kết quả

### LSTM (đã có)
- Accuracy: ~93%
- F1-Score: ~0.92
- Train time: 20-30 phút

### Transformer (dự kiến)
- Accuracy: **94-96%** (nếu tune tốt)
- F1-Score: ~0.93-0.95
- Train time: 15-25 phút (nhanh hơn do parallel)

---

## 🔧 Troubleshooting

### 1. Out of Memory (GPU)
```python
# Giảm batch size
BATCH_SIZE = 8  # thay vì 16

# Giảm d_model
D_MODEL = 64  # thay vì 128
```

### 2. Overfit (Val Acc thấp hơn Train Acc nhiều)
```python
# Tăng dropout
DROPOUT = 0.3

# Giảm model complexity
NUM_LAYERS = 2
D_MODEL = 128
```

### 3. Underfit (Cả Train và Val Acc đều thấp)
```python
# Tăng model capacity
D_MODEL = 256
NUM_LAYERS = 4
NHEAD = 8

# Train lâu hơn
EPOCHS = 100
```

### 4. Training quá chậm
```python
# Tăng batch size (nếu GPU đủ)
BATCH_SIZE = 32

# Giảm complexity
NUM_LAYERS = 2
```

---

## 📊 Visualization & Analysis

### 1. Attention Weights Visualization

Để hiểu Transformer học gì:

```python
from model_transformer import CropClassifierTransformer

# Load model
model = CropClassifierTransformer(...)
model.load_state_dict(...)

# Extract attention (cần implement hook)
# TODO: Thêm hàm get_attention_weights() để visualize
```

### 2. Temporal Pattern Analysis

So sánh cách LSTM vs Transformer xử lý time series:
- LSTM: Sequential, từng bước
- Transformer: Attention toàn bộ chuỗi cùng lúc

---

## 📝 Viết Báo cáo Luận văn

### Phần Methodology

**LSTM Architecture:**
> "Mô hình CNN-LSTM sử dụng CNN để trích xuất đặc trưng không gian từ mỗi time step, 
> sau đó Bi-LSTM học patterns temporal. Global average pooling aggregates features 
> từ toàn bộ chuỗi thời gian trước khi phân loại."

**Transformer Architecture:**
> "Mô hình CNN-Transformer thay thế LSTM bằng Transformer encoder với multi-head 
> self-attention mechanism. Positional encoding được thêm vào để model nhận biết 
> thông tin thứ tự thời gian. Multi-head attention cho phép model tập trung vào 
> các time steps quan trọng khác nhau, cải thiện khả năng học long-range dependencies."

### Phần Results

1. **Table so sánh metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Training time, Parameters count

2. **Confusion Matrices** side-by-side

3. **Per-class Performance** bar chart

4. **Ablation Study** (nếu có thời gian):
   - Số layers ảnh hưởng như thế nào?
   - NHEAD tối ưu là bao nhiêu?
   - Có cần positional encoding không?

### Phần Discussion

- Tại sao Transformer tốt hơn (hoặc không)?
- Phân tích confusion pairs
- Trade-off giữa complexity và performance
- Khuyến nghị cho real-world deployment

---

## 🎓 Advanced Extensions

### 1. Hybrid Model (CNN + LSTM + Transformer)
```python
class HybridModel(nn.Module):
    def __init__(self):
        # CNN → LSTM → Transformer → Classifier
        pass
```

### 2. Temporal Attention Pooling
Thay vì average pooling, dùng attention để weight các time steps.

### 3. Multi-Scale Temporal Features
Dùng nhiều kernel sizes khác nhau trong CNN.

### 4. Data Augmentation
- Random time shift
- Random crop
- Mixup

---

## 📚 Tài liệu Tham khảo

1. **Transformer Original Paper:**
   - Vaswani et al. "Attention Is All You Need" (2017)

2. **Time Series with Transformers:**
   - "Temporal Fusion Transformers" (2019)
   - "Informer: Beyond Efficient Transformer" (2021)

3. **Remote Sensing:**
   - "Satellite Image Time Series Classification" papers

---

## 💡 Tips cho Luận văn

1. **Làm rõ đóng góp:**
   - So sánh 2 architectures state-of-the-art
   - Phân tích sâu về temporal learning
   - Đề xuất deployment strategy

2. **Ablation study quan trọng:**
   - Chứng minh từng component có tác dụng
   - Justification cho design choices

3. **Visualize attention:**
   - Chứng minh model "hiểu" temporal patterns
   - Giải thích interpretability

4. **Real-world considerations:**
   - Inference speed
   - Memory requirements
   - Scalability

---

## ✅ Checklist Hoàn thành

- [ ] Train LSTM baseline (đã xong - 93%)
- [ ] Train Transformer
- [ ] So sánh 2 models
- [ ] Visualize attention weights
- [ ] Ablation study
- [ ] Viết báo cáo methodology
- [ ] Tạo presentation slides
- [ ] Demo application (optional)

---

## 📧 Support

Nếu có vấn đề, kiểm tra:
1. Python version >= 3.8
2. PyTorch version >= 1.10
3. CUDA available (cho GPU training)
4. Đủ RAM (ít nhất 8GB)

Good luck với luận văn! 🎓🚀
