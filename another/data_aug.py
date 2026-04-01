import os
import numpy as np
import json
import rasterio
import shutil

# ================= CAU HINH TANG CUONG MOI =================
DATA_DIR = "dataset_final/train" 

# Muc tieu: Dua tat ca ve nguong ~400-450 mau de do lech voi Lua
AUGMENT_CONFIG = {
    # 'mia': 18,    # 24 goc * 18 = 432 mau
    # 'chanh': 7,   # 65 goc * 7 = 455 mau
    'sen': 2,      # 76 goc * 6 = 456 mau
    # 'dua': 1*2     # 206 goc * 2 = 412 mau
    # 'lua': 0    # Lua 1178 giu nguyen
}

def add_noise(data, noise_level=0.02):
    """Them nhieu"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, sigma=0.04):
    """Co gian"""
    factor = np.random.normal(1.0, sigma)
    return data * factor

def shift_values(data, shift_max=0.03):
    """Dich chuyen"""
    shift = np.random.uniform(-shift_max, shift_max)
    return data + shift

def process_augmentation():
    print(f"--- Bat dau tang cuong du lieu can bang voi Lua ---")
    print(f"Thu muc: {DATA_DIR}")
    
    for crop_type, factor in AUGMENT_CONFIG.items():
        crop_path = os.path.join(DATA_DIR, crop_type)
        if not os.path.exists(crop_path):
            continue

        # Lay danh sach mau goc (bo qua mau da augment truoc do neu co)
        polygons = [d for d in os.listdir(crop_path) 
                   if os.path.isdir(os.path.join(crop_path, d)) and "_aug_" not in d]
        
        print(f"\nLoai cay {crop_type.upper()}: {len(polygons)} goc x {factor} lan = ~{len(polygons)*factor} mau gia")
        
        count_created = 0
        for poly_name in polygons:
            src_poly_dir = os.path.join(crop_path, poly_name)
            
            # Doc metadata
            meta_path = os.path.join(src_poly_dir, 'metadata.json')
            if not os.path.exists(meta_path): continue
            
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_original = json.load(f)

            for i in range(factor):
                # Tao ten file moi
                new_poly_name = f"{poly_name}_aug_v2_{i+1}" # Them v2 de khong trung cai cu
                new_poly_dir = os.path.join(crop_path, new_poly_name)
                
                if os.path.exists(new_poly_dir): continue
                os.makedirs(new_poly_dir)
                
                # Copy metadata
                with open(os.path.join(new_poly_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(meta_original, f, indent=2)
                
                # Bien doi anh
                for fname in os.listdir(src_poly_dir):
                    if fname.endswith('.tif'):
                        with rasterio.open(os.path.join(src_poly_dir, fname)) as src:
                            data = src.read()
                            profile = src.profile
                        
                        # Random ky thuat bien doi
                        rand = np.random.rand()
                        if rand < 0.33: new_data = add_noise(data)
                        elif rand < 0.66: new_data = scale_data(data)
                        else: new_data = shift_values(data)
                            
                        with rasterio.open(os.path.join(new_poly_dir, fname), 'w', **profile) as dst:
                            dst.write(new_data.astype(rasterio.float32))
                
                count_created += 1
        
        print(f"   OK: Da tao xong {count_created} mau cho {crop_type}")

    print("\n--- DA CAN BANG DU LIEU XONG ---")

if __name__ == "__main__":
    process_augmentation()