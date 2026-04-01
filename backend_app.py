"""
Flask Backend cho Web App Phân loại Cây trồng
- API để nhận polygon từ frontend
- Download ảnh vệ tinh từ Google Earth Engine hoặc Sentinel Hub
- Extract features từ time series
- Dự đoán bằng Transformer model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import ee
import json
from datetime import datetime, timedelta
import os

# Import model
from model_transformer import CropClassifierTransformer

app = Flask(__name__)
CORS(app)  # Enable CORS cho Vue.js frontend

# ================= CẤU HÌNH =================
MODEL_PATH = "best_model_transformer.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Earth Engine credentials (cần setup)
# ee.Authenticate()  # Chạy 1 lần để authenticate
PROJECT_ID = 'marine-access-482003-b1'

ee.Initialize(project=PROJECT_ID)

# Model config
INPUT_DIM = 9
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3

# Class names - CẬP NHẬT theo model của bạn
CLASS_NAMES = ['dua', 'lua', 'mia', 'tram']
NUM_CLASSES = len(CLASS_NAMES)

# ================= LOAD MODEL =================
print("🔄 Loading Transformer model...")
model = CropClassifierTransformer(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ Model loaded successfully!")


# ================= HELPER FUNCTIONS =================

def download_sentinel_data(polygon_coords, start_date, end_date):
    """
    Download Sentinel-2 data cho polygon
    
    Args:
        polygon_coords: List of [lon, lat] coordinates
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        
    Returns:
        time_series: (T, 9) numpy array với 9 bands
    """
    try:
        # Tạo polygon geometry
        polygon = ee.Geometry.Polygon(polygon_coords)
        
        # Load Sentinel-2 collection
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(polygon) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Function để tính NDVI, EVI, NDWI
        def add_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4'),
                    'BLUE': image.select('B2')
                }
            ).rename('EVI')
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            return image.addBands([ndvi, evi, ndwi])
        
        # Apply indices
        collection = collection.map(add_indices)
        
        # Select bands: B2, B3, B4, B8, B11, B12, NDVI, EVI, NDWI
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI']
        
        # Get time series
        time_series_list = []
        
        # Get list of images
        image_list = collection.toList(collection.size())
        n_images = image_list.size().getInfo()
        
        print(f"  Found {n_images} images")
        
        if n_images == 0:
            return None
        
        for i in range(min(n_images, 15)):  # Lấy tối đa 15 images
            image = ee.Image(image_list.get(i))
            
            # Reduce to mean values over polygon
            stats = image.select(bands).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=10,  # 10m resolution
                maxPixels=1e9
            ).getInfo()
            
            # Extract values
            values = [stats.get(band, 0) for band in bands]
            time_series_list.append(values)
        
        time_series = np.array(time_series_list)
        
        return time_series
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None


def preprocess_time_series(time_series, target_length=10):
    """
    Preprocess time series to match model input
    
    Args:
        time_series: (T, 9) array
        target_length: desired length
        
    Returns:
        processed: (1, target_length, 9) tensor
    """
    # Handle NaN values
    time_series = np.nan_to_num(time_series, nan=0.0)
    
    current_len = len(time_series)
    
    # Pad or truncate
    if current_len > target_length:
        # Truncate
        processed = time_series[:target_length]
    elif current_len < target_length:
        # Pad with zeros
        pad_len = target_length - current_len
        padding = np.zeros((pad_len, 9))
        processed = np.vstack([time_series, padding])
    else:
        processed = time_series
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
    
    return tensor


def predict_crop(time_series_tensor):
    """
    Predict crop type using Transformer model
    
    Args:
        time_series_tensor: (1, T, 9) tensor
        
    Returns:
        prediction: class name
        confidence: probability
        all_probs: probabilities for all classes
    """
    with torch.no_grad():
        time_series_tensor = time_series_tensor.to(DEVICE)
        
        # Forward pass
        outputs = model(time_series_tensor)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Get prediction
        confidence, predicted = torch.max(probs, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_val = confidence.item()
        
        # All probabilities
        all_probs = {
            CLASS_NAMES[i]: float(probs[0][i].item()) 
            for i in range(NUM_CLASSES)
        }
    
    return predicted_class, confidence_val, all_probs


# ================= API ENDPOINTS =================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Transformer',
        'device': str(DEVICE),
        'classes': CLASS_NAMES
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Request body:
    {
        "polygon": [[lon1, lat1], [lon2, lat2], ...],
        "start_date": "2024-01-01",
        "end_date": "2024-06-01"
    }
    
    Response:
    {
        "success": true,
        "prediction": "lua",
        "confidence": 0.95,
        "probabilities": {"lua": 0.95, "mia": 0.03, ...},
        "n_images": 12
    }
    """
    try:
        data = request.json
        
        # Validate input
        if 'polygon' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing polygon coordinates'
            }), 400
        
        polygon = data['polygon']
        start_date = data.get('start_date', '2024-01-01')
        end_date = data.get('end_date', '2024-06-01')
        
        print(f"\n📍 Received prediction request")
        print(f"  Polygon: {len(polygon)} points")
        print(f"  Date range: {start_date} to {end_date}")
        
        # Download satellite data
        print("📡 Downloading satellite data...")
        time_series = download_sentinel_data(polygon, start_date, end_date)
        
        if time_series is None or len(time_series) == 0:
            return jsonify({
                'success': False,
                'error': 'No satellite data found for this location and time range'
            }), 404
        
        print(f"✓ Downloaded {len(time_series)} images")
        
        # Preprocess
        print("🔧 Preprocessing data...")
        time_series_tensor = preprocess_time_series(time_series)
        
        # Predict
        print("🤖 Running prediction...")
        prediction, confidence, all_probs = predict_crop(time_series_tensor)
        
        print(f"✓ Prediction: {prediction} ({confidence*100:.2f}%)")
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': all_probs,
            'n_images': len(time_series),
            'message': f'Dự đoán: {prediction} với độ tin cậy {confidence*100:.1f}%'
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-mock', methods=['POST'])
def predict_mock():
    """
    Mock endpoint for testing without downloading real data
    Useful khi chưa setup Google Earth Engine
    """
    try:
        data = request.json
        
        # Generate fake time series
        np.random.seed(42)
        time_series = np.random.randn(10, 9) * 0.1 + 0.5
        time_series = np.clip(time_series, 0, 1)
        
        # Preprocess
        time_series_tensor = preprocess_time_series(time_series)
        
        # Predict
        prediction, confidence, all_probs = predict_crop(time_series_tensor)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': all_probs,
            'n_images': 10,
            'message': f'[MOCK] Dự đoán: {prediction} ({confidence*100:.1f}%)',
            'is_mock': True
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of supported crop classes"""
    return jsonify({
        'classes': CLASS_NAMES,
        'num_classes': NUM_CLASSES
    })


# ================= MAIN =================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌾 CROP CLASSIFICATION API SERVER")
    print("="*70)
    print(f"Model: Transformer")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Device: {DEVICE}")
    print("="*70 + "\n")
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )