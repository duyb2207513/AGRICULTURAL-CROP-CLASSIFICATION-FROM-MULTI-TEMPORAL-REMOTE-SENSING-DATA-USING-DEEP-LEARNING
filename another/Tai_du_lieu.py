import geopandas as gpd
import ee
import os
import json
import requests
import datetime
import time
from tqdm import tqdm
import rasterio
from rasterio.mask import mask as rasterio_mask
import numpy as np
from shapely.geometry import mapping
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==============================================================================
# CAU HINH TOI UU CHO RAM 8GB
# ==============================================================================
PROJECT_ID = 'marine-access-482003-b1'

# 1. Tro vao file Shapefile MOI
SHP_PATH = "LuanVan/taive_2_2.shp" 

# 2. Luu ra thu muc tam de kiem tra
OUTPUT_BASE_DIR = "dataset_bo_sung" 

REGION_NAME = "Chau_Thanh_Soc_Trang"
TEST_LIMIT = 0

# 3. Danh sach loai cay tai ve
CROPS_TO_DOWNLOAD = ['lua','sen', 'mia', 'dua', 'tram']

# Cau hinh thoi gian
CROP_TIME_CONFIG = {
    'lua': {'start_date': '2023-12-15', 'end_date': '2024-5-1', 'time_step_days': 15, 'min_images': 9},
    'dua': {'start_date': '2023-12-15', 'end_date': '2024-5-1', 'time_step_days': 15, 'min_images': 9},
    'tram': {'start_date': '2023-12-15', 'end_date': '2024-5-1', 'time_step_days': 15, 'min_images': 9},
    'sen': {'start_date': '2023-12-15', 'end_date': '2024-5-1', 'time_step_days': 15, 'min_images': 9},
    'mia': {'start_date': '2023-12-15', 'end_date': '2024-5-1', 'time_step_days': 15, 'min_images': 9}
}

LABEL_MAPPING = {
    'sen': 'sen', 'lotus': 'sen',
    'mia': 'mia', 'sugarcane': 'mia',
    'dua': 'dua', 'coconut': 'dua',
    'tram': 'tram', 'tangerine': 'tram',
}

SCALE = 10
CLOUD_THRESHOLD = 40
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI']

# MULTI-THREADING
MAX_WORKERS = 8
REQUEST_DELAY = 0.1

# Lock cho thread-safe
stats_lock = threading.Lock()

# Khoi tao GEE
try:
    ee.Initialize(project=PROJECT_ID)
    print("OK: GEE da duoc khoi tao")
except:
    print("Xac thuc: Dang xac thuc GEE...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("OK: GEE da duoc khoi tao")

def mask_clouds_sentinel2(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
    ).rename('EVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, evi, ndwi]).select(BANDS)

def normalize_label(label):
    label_lower = str(label).lower().strip()
    return LABEL_MAPPING.get(label_lower, 'khac')

def generate_timeline(start_str, end_str, step_days):
    start = datetime.datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_str, '%Y-%m-%d')
    timeline = []
    current = start
    while current < end:
        next_step = current + datetime.timedelta(days=step_days)
        if next_step > end: next_step = end
        timeline.append((current.strftime('%Y-%m-%d'), next_step.strftime('%Y-%m-%d')))
        current = next_step
    return timeline

def split_dataset(gdf, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    crop_counts = gdf['crop_type'].value_counts()
    print("\n--- Phan bo nhan ---")
    for crop, count in crop_counts.items():
        print(f"   {crop}: {count} mau")
    
    train_gdf, temp_gdf = train_test_split(
        gdf, test_size=(val_ratio + test_ratio),
        stratify=gdf['crop_type'], random_state=random_state
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_gdf, test_gdf = train_test_split(
        temp_gdf, test_size=(1 - val_size),
        stratify=temp_gdf['crop_type'], random_state=random_state
    )
    
    print(f"\nChia du lieu:")
    print(f"   Train: {len(train_gdf)} mau ({len(train_gdf)/len(gdf)*100:.1f}%)")
    print(f"   Val:   {len(val_gdf)} mau ({len(val_gdf)/len(gdf)*100:.1f}%)")
    print(f"   Test:  {len(test_gdf)} mau ({len(test_gdf)/len(gdf)*100:.1f}%)")
    
    return train_gdf, val_gdf, test_gdf

def download_single_polygon(args):
    idx, row, timeline, split_name, crop_type = args
    poly_folder = os.path.join(OUTPUT_BASE_DIR, split_name, crop_type, f"poly_{idx}")
    os.makedirs(poly_folder, exist_ok=True)
    
    geom = row.geometry
    geom_json = mapping(geom)
    ee_geometry = ee.Geometry(geom_json)
    
    meta = {
        "polygon_id": int(idx),
        "crop_type": crop_type,
        "original_label": row.get('label', 'unknown'),
        "area_m2": float(geom.area),
        "bounds": list(geom.bounds),
        "split": split_name,
        "timeline": []
    }
    
    images_downloaded = 0
    success = 0
    fail = 0
    
    for t_idx, (t_start, t_end) in enumerate(timeline):
        file_name = f"t{t_idx:02d}_{t_start}.tif"
        file_path = os.path.join(poly_folder, file_name)
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            meta["timeline"].append({"step": t_idx, "date": t_start, "status": "exists"})
            images_downloaded += 1
            continue
        
        try:
            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterDate(t_start, t_end)
                   .filterBounds(ee_geometry)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
                   .map(mask_clouds_sentinel2)
                   .map(add_indices))
            
            if col.size().getInfo() > 0:
                img = col.median().clip(ee_geometry)
                bounds = geom.bounds
                region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
                
                url = img.getDownloadURL({
                    'region': region.getInfo()['coordinates'],
                    'scale': SCALE, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'
                })
                
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200:
                    temp_path = file_path + ".temp"
                    with open(temp_path, 'wb') as f: f.write(resp.content)
                    
                    with rasterio.open(temp_path) as src:
                        out_image, out_transform = rasterio_mask(src, [geom], crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff", "height": out_image.shape[1],
                            "width": out_image.shape[2], "transform": out_transform
                        })
                        with rasterio.open(file_path, "w", **out_meta) as dest: dest.write(out_image)
                    os.remove(temp_path)
                    meta["timeline"].append({"step": t_idx, "date": t_start, "status": "success"})
                    images_downloaded += 1
                    success += 1
                else:
                    meta["timeline"].append({"step": t_idx, "status": "http_error"})
                    fail += 1
            else:
                meta["timeline"].append({"step": t_idx, "status": "no_data"})
                fail += 1
        except Exception as e:
            meta["timeline"].append({"step": t_idx, "status": "error", "msg": str(e)})
            fail += 1
        time.sleep(REQUEST_DELAY)
    
    meta["total_images"] = images_downloaded
    meta["is_valid"] = images_downloaded >= CROP_TIME_CONFIG[crop_type]['min_images']
    with open(os.path.join(poly_folder, "metadata.json"), "w", encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    return {'success': success, 'fail': fail, 'total_images': images_downloaded}

def download_parallel(gdf, split_name='train'):
    print(f"\n--- THU THAP SONG SONG - {split_name.upper()} ---")
    if TEST_LIMIT and TEST_LIMIT > 0:
        print(f"CANH BAO: Che do TEST, chi tai {TEST_LIMIT} mau moi loai!")
    
    tasks = []
    for crop_type in gdf['crop_type'].unique():
        if crop_type not in CROP_TIME_CONFIG: continue
        crop_gdf = gdf[gdf['crop_type'] == crop_type]
        if TEST_LIMIT and TEST_LIMIT > 0:
            crop_gdf = crop_gdf.head(TEST_LIMIT)
        
        timeline = generate_timeline(
            CROP_TIME_CONFIG[crop_type]['start_date'],
            CROP_TIME_CONFIG[crop_type]['end_date'],
            CROP_TIME_CONFIG[crop_type]['time_step_days']
        )
        for idx, row in crop_gdf.iterrows():
            tasks.append((idx, row, timeline, split_name, crop_type))
    
    if len(tasks) == 0: return {'success': 0, 'fail': 0, 'total_images': 0}

    total_stats = {'success': 0, 'fail': 0, 'total_images': 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_polygon, task): task for task in tasks}
        with tqdm(total=len(tasks), desc=f"{split_name}", unit="polygon") as pbar:
            for future in as_completed(futures):
                result = future.result()
                with stats_lock:
                    total_stats['success'] += result['success']
                    total_stats['fail'] += result['fail']
                    total_stats['total_images'] += result['total_images']
                pbar.update(1)
    return total_stats

def validate_dataset(base_dir):
    print(f"\n--- KIEM TRA CHAT LUONG DATASET ---")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir): continue
        print(f"\nTAP {split.upper()}:")
        for crop_type in os.listdir(split_dir):
            crop_dir = os.path.join(split_dir, crop_type)
            if not os.path.isdir(crop_dir): continue
            polygons = [p for p in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, p))]
            total_images = 0
            valid_polygons = 0
            for poly in polygons:
                poly_dir = os.path.join(crop_dir, poly)
                total_images += len([f for f in os.listdir(poly_dir) if f.endswith('.tif')])
                meta_file = os.path.join(poly_dir, 'metadata.json')
                if os.path.exists(meta_file):
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        if json.load(f).get('is_valid', False): valid_polygons += 1
            avg = total_images / len(polygons) if polygons else 0
            print(f"   {crop_type:15s}: {len(polygons):4d} poly, {total_images:5d} anh, TB: {avg:.1f}, Hop le: {valid_polygons}/{len(polygons)}")

def create_dataset_info(base_dir, train_gdf, val_gdf, test_gdf):
    dataset_info = {
        "name": "Crop Classification Multi-temporal Dataset",
        "bands": BANDS,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    os.makedirs(base_dir, exist_ok=True) 
    with open(os.path.join(base_dir, "dataset_info.json"), "w", encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"Da luu dataset_info.json")

if __name__ == "__main__":
    print("--- THU THAP DU LIEU SONG SONG ---")
    if not os.path.exists(SHP_PATH):
        print(f"LOI: Khong tim thay {SHP_PATH}")
        exit()
    
    gdf = gpd.read_file(SHP_PATH)
    gdf = gdf[gdf.geometry.notnull()]
    gdf['crop_type'] = gdf['label'].apply(normalize_label)
    gdf = gdf[gdf['crop_type'].isin(CROPS_TO_DOWNLOAD)]
    
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    
    train_gdf, val_gdf, test_gdf = split_dataset(gdf)
    create_dataset_info(OUTPUT_BASE_DIR, train_gdf, val_gdf, test_gdf)
    
    user_input = input("Bat dau thu thap song song? (y/n): ")
    if user_input.lower() == 'y':
        start_time = time.time()
        download_parallel(train_gdf, 'train')
        download_parallel(val_gdf, 'val')
        download_parallel(test_gdf, 'test')
        validate_dataset(OUTPUT_BASE_DIR)
        print(f"\nThoi gian thuc hien: {(time.time() - start_time) / 3600:.2f} gio")
        print("HOAN THANH!")
    else:
        print("Huy bo.")