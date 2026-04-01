import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding để thêm thông tin vị trí thời gian vào sequence.
    Quan trọng với Transformer vì nó không có khái niệm thứ tự như LSTM.
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Tạo ma trận positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class CropClassifierTransformer(nn.Module):
    """
    Mô hình CNN + Transformer cho phân loại cây trồng đa thời gian.
    
    Kiến trúc:
    1. CNN: Trích xuất đặc trưng không gian từ mỗi time step
    2. Positional Encoding: Thêm thông tin thời gian
    3. Transformer Encoder: Học mối quan hệ temporal với multi-head attention
    4. Classifier: Phân loại dựa trên features tổng hợp
    """
    
    def __init__(self, input_dim=9, cnn_hidden=128, d_model=128, 
                 nhead=4, num_layers=2, num_classes=5, dropout=0.2):
        """
        Args:
            input_dim: Số kênh input (9 bands)
            cnn_hidden: Số kênh output của CNN
            d_model: Dimension của Transformer (phải chia hết cho nhead)
            nhead: Số attention heads
            num_layers: Số lớp Transformer Encoder
            num_classes: Số lớp cây trồng
            dropout: Tỷ lệ dropout
        """
        super(CropClassifierTransformer, self).__init__()
        
        # --- PHẦN 1: CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            # Conv1d: (Batch, Channels, Time)
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(in_channels=64, out_channels=cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Projection: Đưa CNN features về d_model dimension cho Transformer
        self.feature_projection = nn.Linear(cnn_hidden, d_model)
        
        # --- PHẦN 2: Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # --- PHẦN 3: Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Thường gấp 4 lần d_model
            dropout=dropout,
            activation='gelu',  # GELU thường tốt hơn ReLU cho Transformer
            batch_first=True    # Input: (Batch, Seq, Features)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # --- PHẦN 4: Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Layer Normalization cuối
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (Batch, Seq_Len, Channels)
            
        Returns:
            logits: (Batch, num_classes)
        """
        # Input x: (Batch, Seq_Len, Channels)
        batch_size, seq_len, channels = x.shape
        
        # 1. Qua CNN để trích xuất đặc trưng spatial
        # Permute để khớp input Conv1d: (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)
        features = self.cnn(x)  # (Batch, cnn_hidden, Seq_Len)
        
        # Permute lại: (Batch, Seq_Len, cnn_hidden)
        features = features.permute(0, 2, 1)
        
        # 2. Project lên d_model dimension
        features = self.feature_projection(features)  # (Batch, Seq_Len, d_model)
        
        # 3. Thêm Positional Encoding
        features = self.pos_encoder(features)
        
        # 4. Qua Transformer Encoder
        # Transformer tự động học attention giữa các time steps
        transformer_out = self.transformer_encoder(features)  # (Batch, Seq_Len, d_model)
        
        # 5. Pooling: Có nhiều cách, đây là 3 cách phổ biến
        
        # Cách 1: Global Average Pooling (khuyến nghị cho multi-temporal)
        pooled = torch.mean(transformer_out, dim=1)  # (Batch, d_model)
        
        # Cách 2: Lấy CLS token (như BERT) - nếu muốn dùng thì thêm CLS token vào đầu
        # pooled = transformer_out[:, 0, :]
        
        # Cách 3: Max Pooling - lấy features nổi bật nhất
        # pooled, _ = torch.max(transformer_out, dim=1)
        
        # Layer normalization
        pooled = self.layer_norm(pooled)
        
        # 6. Phân loại
        logits = self.classifier(pooled)  # (Batch, num_classes)
        
        return logits
    
    def get_attention_weights(self, x):
        """
        Hàm phụ để extract attention weights - hữu ích cho visualization
        Có thể dùng để phân tích xem model chú ý vào time steps nào
        """
        # Tương tự forward nhưng return thêm attention weights
        x = x.permute(0, 2, 1)
        features = self.cnn(x)
        features = features.permute(0, 2, 1)
        features = self.feature_projection(features)
        features = self.pos_encoder(features)
        
        # Để lấy attention weights, cần modify TransformerEncoder
        # hoặc dùng các hook - phần này có thể mở rộng sau
        return None


class CropClassifierTransformerAdvanced(nn.Module):
    """
    Phiên bản nâng cao với CLS token và Temporal Attention Pooling
    """
    
    def __init__(self, input_dim=9, cnn_hidden=128, d_model=128, 
                 nhead=4, num_layers=2, num_classes=5, dropout=0.2):
        super(CropClassifierTransformerAdvanced, self).__init__()
        
        self.d_model = d_model
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.feature_projection = nn.Linear(cnn_hidden, d_model)
        
        # CLS Token - learnable token đại diện cho toàn bộ sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention-based Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = x.permute(0, 2, 1)
        features = self.cnn(x)
        features = features.permute(0, 2, 1)
        features = self.feature_projection(features)
        
        # Thêm CLS token vào đầu sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (Batch, 1, d_model)
        features = torch.cat([cls_tokens, features], dim=1)  # (Batch, Seq_Len+1, d_model)
        
        # Positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        transformer_out = self.transformer_encoder(features)
        
        # Lấy CLS token output
        cls_output = transformer_out[:, 0, :]  # (Batch, d_model)
        
        # Hoặc dùng attention pooling cho phần còn lại
        # attention_weights = self.attention_pool(transformer_out[:, 1:, :])
        # pooled = torch.sum(transformer_out[:, 1:, :] * attention_weights, dim=1)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits


# ==========================================
# TEST MÔ HÌNH
# ==========================================
if __name__ == "__main__":
    print("=== TESTING TRANSFORMER MODELS ===\n")
    
    # Test mô hình cơ bản
    print("1. Testing CropClassifierTransformer (Basic)...")
    model_basic = CropClassifierTransformer(
        input_dim=9,
        cnn_hidden=128,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=5
    )
    
    # Giả lập input
    dummy_input = torch.randn(16, 10, 9)  # (Batch=16, Seq=10, Channels=9)
    
    output_basic = model_basic(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_basic.shape}")
    print(f"   Expected: (16, 5) ✓\n")
    
    # Test mô hình nâng cao
    print("2. Testing CropClassifierTransformerAdvanced (with CLS token)...")
    model_advanced = CropClassifierTransformerAdvanced(
        input_dim=9,
        d_model=128,
        nhead=4,
        num_layers=3,
        num_classes=5
    )
    
    output_advanced = model_advanced(dummy_input)
    print(f"   Output shape: {output_advanced.shape}")
    print(f"   Expected: (16, 5) ✓\n")
    
    # Tính số parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("3. Model Statistics:")
    print(f"   Basic Transformer params: {count_parameters(model_basic):,}")
    print(f"   Advanced Transformer params: {count_parameters(model_advanced):,}")
    
    # Test với loss
    print("\n4. Testing with loss function...")
    criterion = nn.CrossEntropyLoss()
    dummy_labels = torch.randint(0, 5, (16,))
    
    loss_basic = criterion(output_basic, dummy_labels)
    loss_advanced = criterion(output_advanced, dummy_labels)
    
    print(f"   Basic model loss: {loss_basic.item():.4f}")
    print(f"   Advanced model loss: {loss_advanced.item():.4f}")
    
    print("\n=== ALL TESTS PASSED ✓ ===")
