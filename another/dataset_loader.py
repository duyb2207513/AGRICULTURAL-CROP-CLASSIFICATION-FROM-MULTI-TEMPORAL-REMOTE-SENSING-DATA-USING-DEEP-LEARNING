import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

# ================= CAU HINH =================
# Danh sach 9 kenh. Bat buoc file TIF dau vao phai co du 9 kenh nay theo thu tu.
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI']

class CropTimeSeriesDataset(Dataset):
    def __init__(self, root_dir, split='train', max_len=10):
        self.root_dir = os.path.join(root_dir, split)
        self.max_len = max_len
        self.samples = []
        
        # 1. Tu dong tim danh sach lop (folder con)
        if os.path.exists(self.root_dir):
            self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        else:
            print(f"LOI: Khong tim thay thu muc: {self.root_dir}")
            self.classes = []
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 2. Quet toan bo mau du lieu
        print(f"--- Dang quet du lieu {split}... ---")
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for poly_id in os.listdir(cls_folder):
                poly_path = os.path.join(cls_folder, poly_id)
                # Chi can la thu muc thi coi la 1 mau
                if os.path.isdir(poly_path):
                    self.samples.append((poly_path, self.class_to_idx[cls_name]))

        print(f"Ket qua: Tim thay {len(self.samples)} mau cho tap {split}. Cac lop: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        poly_path, label = self.samples[idx]
        
        # 1. Lay danh sach file anh va sap xep theo thoi gian (ten file)
        files = sorted([f for f in os.listdir(poly_path) if f.endswith('.tif')])
        
        time_series_data = []
        
        # 2. Doc tung file anh
        for fname in files:
            img_path = os.path.join(poly_path, fname)
            try:
                with rasterio.open(img_path) as src:
                    # Kiem tra so luong band
                    if src.count != len(BANDS):
                        time_series_data.append(np.full(len(BANDS), np.nan))
                        continue
                        
                    # Doc du lieu: (Bands, H, W)
                    img_data = src.read()
                    
                    # Tinh trung binh khong gian (Global Average Pooling) -> (Bands,)
                    mean_val = np.nanmean(img_data, axis=(1, 2))
                    
                    # Neu toan bo la NaN (anh rong hoan toan)
                    if np.isnan(mean_val).any():
                        time_series_data.append(np.full(len(BANDS), np.nan))
                    else:
                        time_series_data.append(mean_val)
            except:
                # Neu file loi khong mo duoc
                time_series_data.append(np.full(len(BANDS), np.nan))
        
        # Chuyen sang Numpy Array: (Time_steps, Bands)
        data_matrix = np.array(time_series_data)
        
        # Neu mau khong co anh nao, tao 1 dong toan 0
        if len(data_matrix) == 0:
             data_matrix = np.zeros((1, len(BANDS)))

        # 3. --- XU LY NOI SUY (INTERPOLATION) ---
        for col in range(data_matrix.shape[1]):
            series = data_matrix[:, col]
            if np.isnan(series).any():
                x = np.arange(len(series))
                idx_good = ~np.isnan(series)
                if np.sum(idx_good) > 0:
                    series = np.interp(x, x[idx_good], series[idx_good])
                    data_matrix[:, col] = series
                else:
                    data_matrix[:, col] = 0.0

        data_matrix = np.nan_to_num(data_matrix)

        # 4. Padding hoac Cat
        current_len = len(data_matrix)
        if current_len > self.max_len:
            data_matrix = data_matrix[:self.max_len]
        elif current_len < self.max_len:
            pad_len = self.max_len - current_len
            padding = np.zeros((pad_len, len(BANDS)))
            data_matrix = np.vstack([data_matrix, padding])

        # 5. Chuyen sang Tensor
        return torch.tensor(data_matrix, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==========================================
# CHAY TEST THU
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "dataset_bo_sung" 
    
    if os.path.exists(DATA_DIR):
        train_ds = CropTimeSeriesDataset(DATA_DIR, split='train', max_len=10)
        
        if len(train_ds) > 0:
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
            data, labels = next(iter(train_loader))
            
            print("\n--- Kiem tra Batch dau tien ---")
            print(f"Data Shape: {data.shape} (Ky vong: Batch, 10, 9)") 
            print(f"Labels: {labels}")
            print("\nData Loader da san sang de train!")
        else:
            print("CANH BAO: Dataset rong! Kiem tra lai duong dan.")
    else:
        print(f"LOI: Khong tim thay thu muc: {DATA_DIR}")