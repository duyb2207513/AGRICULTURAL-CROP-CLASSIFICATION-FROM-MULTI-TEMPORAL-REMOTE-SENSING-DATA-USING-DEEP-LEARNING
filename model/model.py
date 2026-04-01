import torch
import torch.nn as nn

class CropClassifier(nn.Module):
    # SỬA 1: input_dim=9 (khớp với dataset), num_classes=5 (khớp số lượng cây)
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, num_classes=5):
        super(CropClassifier, self).__init__()
        
        # --- PHẦN 1: CNN (Feature Extractor) ---
        self.cnn = nn.Sequential(
            # Conv1d: (Batch, Channels, Time)
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        # --- PHẦN 2: LSTM (Sequence Modeling) ---
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True 
        )
        
        # --- PHẦN 3: CLASSIFIER ---
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input x: (Batch, Seq_Len, Channels)
        
        # 1. Qua CNN
        # Permute để khớp input Conv1d: (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1) 
        features = self.cnn(x) 
        
        # 2. Qua LSTM
        # Permute lại để khớp input LSTM: (Batch, Seq_Len, Channels)
        features = features.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(features)
   
        # Cách mới (An toàn): Global Average Pooling
        # Tính trung bình dọc theo trục thời gian (dim=1)
        # Ý nghĩa: "Nhìn tổng quát cả vụ mùa xem trung bình nó giống cây gì"
        avg_pool = torch.mean(lstm_out, dim=1) 
        
        # Hoặc dùng Max Pooling (Lấy đặc điểm nổi bật nhất)
        # max_pool, _ = torch.max(lstm_out, dim=1)
        
        # 3. Phân loại
        logits = self.fc(avg_pool)
        
        return logits

# --- TEST THỬ ---
if __name__ == "__main__":
    # Khai báo model với 9 features, 5 lớp
    model = CropClassifier(input_dim=9, num_classes=5)
    
    # Giả lập input: Batch=32, Dài=10 ngày, 9 kênh màu
    dummy_input = torch.randn(32, 10, 9)
    
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape) # Phải ra (32, 5)
    
    # Kiểm tra xem có chạy được Loss không
    criterion = nn.CrossEntropyLoss()
    dummy_target = torch.randint(0, 5, (32,)) # Nhãn giả ngẫu nhiên 0-4
    loss = criterion(output, dummy_target)
    print("Loss test:", loss.item())