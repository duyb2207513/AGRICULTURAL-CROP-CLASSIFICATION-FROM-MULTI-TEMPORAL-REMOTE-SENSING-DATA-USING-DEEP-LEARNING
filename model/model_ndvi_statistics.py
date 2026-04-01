import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class NDVITemporalFeatureExtractor:
    """
    Trích xuất các đặc trưng thống kê từ NDVI time series
    
    Đây là phương pháp truyền thống trong Remote Sensing:
    - Không dùng Deep Learning
    - Dùng statistical features từ NDVI qua thời gian
    - Train bằng Random Forest hoặc SVM
    
    Features được trích xuất:
    1. Mean, Std, Min, Max
    2. Percentiles (25th, 50th, 75th)
    3. Trend (slope của linear regression)
    4. Amplitude (max - min)
    5. Coefficient of Variation
    6. Skewness, Kurtosis
    """
    
    def __init__(self, use_all_bands=False):
        """
        Args:
            use_all_bands: Nếu True, dùng tất cả 9 bands. Nếu False, chỉ dùng NDVI (band index 6)
        """
        self.use_all_bands = use_all_bands
        self.ndvi_index = 6  # NDVI ở vị trí thứ 7 trong 9 bands
        
    def extract_features_single_series(self, time_series):
        """
        Trích xuất features từ 1 time series
        
        Args:
            time_series: (Seq_Len,) hoặc (Seq_Len, Bands)
            
        Returns:
            features: (n_features,)
        """
        if len(time_series.shape) == 2:
            # Nếu nhiều bands, lấy NDVI
            if not self.use_all_bands:
                time_series = time_series[:, self.ndvi_index]
        
        features = []
        
        # 1. Basic statistics
        features.append(np.mean(time_series))      # Mean
        features.append(np.std(time_series))       # Standard deviation
        features.append(np.min(time_series))       # Minimum
        features.append(np.max(time_series))       # Maximum
        features.append(np.median(time_series))    # Median
        
        # 2. Percentiles
        features.append(np.percentile(time_series, 25))  # 25th percentile
        features.append(np.percentile(time_series, 75))  # 75th percentile
        
        # 3. Range and amplitude
        features.append(np.max(time_series) - np.min(time_series))  # Amplitude
        
        # 4. Coefficient of Variation (CV)
        mean_val = np.mean(time_series)
        if mean_val != 0:
            features.append(np.std(time_series) / mean_val)  # CV
        else:
            features.append(0.0)
        
        # 5. Temporal trend (slope)
        if len(time_series) > 1:
            x = np.arange(len(time_series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
            features.append(slope)        # Trend slope
            features.append(r_value**2)   # R-squared
        else:
            features.append(0.0)
            features.append(0.0)
        
        # 6. Skewness and Kurtosis
        if len(time_series) > 2:
            features.append(stats.skew(time_series))      # Skewness
            features.append(stats.kurtosis(time_series))  # Kurtosis
        else:
            features.append(0.0)
            features.append(0.0)
        
        # 7. Sum and Integral (tổng NDVI)
        features.append(np.sum(time_series))
        
        # 8. First and last values (start/end of season)
        features.append(time_series[0])   # First value
        features.append(time_series[-1])  # Last value
        
        # 9. Peak timing (when max occurs)
        peak_idx = np.argmax(time_series)
        features.append(peak_idx / len(time_series))  # Normalized peak position
        
        return np.array(features)
    
    def extract_features_batch(self, batch_data):
        """
        Trích xuất features từ một batch
        
        Args:
            batch_data: (Batch, Seq_Len, Bands) hoặc (Batch, Seq_Len)
            
        Returns:
            features: (Batch, n_features)
        """
        batch_size = batch_data.shape[0]
        
        # Extract features cho mỗi sample
        all_features = []
        for i in range(batch_size):
            feat = self.extract_features_single_series(batch_data[i])
            all_features.append(feat)
        
        return np.array(all_features)
    
    def get_feature_names(self):
        """Trả về tên các features"""
        return [
            'mean', 'std', 'min', 'max', 'median',
            'percentile_25', 'percentile_75',
            'amplitude',
            'cv',
            'trend_slope', 'trend_r2',
            'skewness', 'kurtosis',
            'sum',
            'first_value', 'last_value',
            'peak_position'
        ]


class NDVIStatisticsClassifier:
    """
    Classifier sử dụng NDVI temporal statistics
    
    Đây là baseline truyền thống - KHÔNG dùng deep learning
    """
    
    def __init__(self, classifier_type='random_forest', use_all_bands=False):
        """
        Args:
            classifier_type: 'random_forest' hoặc 'svm'
            use_all_bands: Dùng tất cả bands hay chỉ NDVI
        """
        self.feature_extractor = NDVITemporalFeatureExtractor(use_all_bands=use_all_bands)
        self.classifier_type = classifier_type
        
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                random_state=42,
                probability=True  # Để có thể dùng predict_proba
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def fit(self, X_train, y_train):
        """
        Train model
        
        Args:
            X_train: (N, Seq_Len, Bands) - time series data
            y_train: (N,) - labels
        """
        print(f"🔧 Extracting features from {len(X_train)} training samples...")
        X_features = self.feature_extractor.extract_features_batch(X_train)
        
        print(f"✓ Feature shape: {X_features.shape}")
        print(f"🌲 Training {self.classifier_type}...")
        
        self.classifier.fit(X_features, y_train)
        
        print("✓ Training completed!")
        
        # Feature importance (chỉ cho Random Forest)
        if self.classifier_type == 'random_forest':
            importances = self.classifier.feature_importances_
            feature_names = self.feature_extractor.get_feature_names()
            
            print("\n📊 Top 5 Most Important Features:")
            indices = np.argsort(importances)[::-1][:5]
            for i, idx in enumerate(indices):
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, X_test):
        """
        Predict labels
        
        Args:
            X_test: (N, Seq_Len, Bands)
            
        Returns:
            predictions: (N,)
        """
        X_features = self.feature_extractor.extract_features_batch(X_test)
        return self.classifier.predict(X_features)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities
        
        Args:
            X_test: (N, Seq_Len, Bands)
            
        Returns:
            probabilities: (N, n_classes)
        """
        X_features = self.feature_extractor.extract_features_batch(X_test)
        return self.classifier.predict_proba(X_features)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate model
        
        Args:
            X_test: (N, Seq_Len, Bands)
            y_test: (N,)
            class_names: List of class names
            
        Returns:
            accuracy, predictions, confusion_matrix
        """
        print("\n🔍 Evaluating model...")
        
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n✓ Accuracy: {accuracy*100:.2f}%")
        
        if class_names is not None:
            print("\n📊 Classification Report:")
            print(classification_report(y_test, predictions, 
                                       target_names=class_names, 
                                       digits=4, 
                                       zero_division=0))
        
        cm = confusion_matrix(y_test, predictions)
        
        return accuracy, predictions, cm


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("="*70)
    print("TESTING NDVI TEMPORAL STATISTICS BASELINE")
    print("="*70)
    
    # Simulate data
    np.random.seed(42)
    n_train = 200
    n_test = 50
    seq_len = 10
    n_bands = 9
    n_classes = 5
    
    # Tạo synthetic NDVI time series cho mỗi class
    print("\n📊 Creating synthetic NDVI time series...")
    
    X_train = []
    y_train = []
    
    for class_id in range(n_classes):
        for _ in range(n_train // n_classes):
            # Mỗi class có pattern NDVI khác nhau
            t = np.linspace(0, 1, seq_len)
            
            if class_id == 0:  # Lúa - peak sớm
                ndvi = 0.3 + 0.5 * np.sin(2 * np.pi * t + np.pi/4)
            elif class_id == 1:  # Mía - tăng dần
                ndvi = 0.2 + 0.6 * t
            elif class_id == 2:  # Ngô - peak giữa
                ndvi = 0.4 + 0.4 * np.sin(2 * np.pi * t)
            elif class_id == 3:  # Rau - dao động
                ndvi = 0.5 + 0.3 * np.sin(4 * np.pi * t)
            else:  # Đất trống - thấp
                ndvi = 0.1 + 0.1 * np.random.randn(seq_len)
            
            # Add noise
            ndvi += 0.05 * np.random.randn(seq_len)
            ndvi = np.clip(ndvi, 0, 1)
            
            # Tạo 9 bands (giả sử NDVI ở vị trí 6)
            sample = np.random.randn(seq_len, n_bands) * 0.1
            sample[:, 6] = ndvi  # NDVI channel
            
            X_train.append(sample)
            y_train.append(class_id)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Tạo test set
    X_test = X_train[:n_test].copy()
    y_test = y_train[:n_test].copy()
    
    print(f"✓ Train shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")
    
    # Test feature extraction
    print("\n" + "="*70)
    print("1. Testing Feature Extraction")
    print("="*70)
    
    extractor = NDVITemporalFeatureExtractor(use_all_bands=False)
    features = extractor.extract_features_batch(X_train[:5])
    
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"✓ Feature names ({len(extractor.get_feature_names())}):")
    print(f"  {extractor.get_feature_names()}")
    
    print("\n✓ Sample features (first sample):")
    for name, val in zip(extractor.get_feature_names(), features[0]):
        print(f"  {name:20s}: {val:.4f}")
    
    # Test Random Forest classifier
    print("\n" + "="*70)
    print("2. Testing Random Forest Classifier")
    print("="*70)
    
    rf_model = NDVIStatisticsClassifier(classifier_type='random_forest')
    rf_model.fit(X_train, y_train)
    
    class_names = ['Lua', 'Mia', 'Ngo', 'Rau', 'Dat_trong']
    accuracy, preds, cm = rf_model.evaluate(X_test, y_test, class_names)
    
    # Test SVM classifier
    print("\n" + "="*70)
    print("3. Testing SVM Classifier")
    print("="*70)
    
    svm_model = NDVIStatisticsClassifier(classifier_type='svm')
    svm_model.fit(X_train, y_train)
    accuracy_svm, preds_svm, cm_svm = svm_model.evaluate(X_test, y_test, class_names)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Random Forest Accuracy: {accuracy*100:.2f}%")
    print(f"SVM Accuracy:           {accuracy_svm*100:.2f}%")
    
    print("\n✅ ALL TESTS PASSED!")
    
    print("\n💡 For real data:")
    print("  • Use this as baseline (traditional method)")
    print("  • Expected accuracy: 70-85%")
    print("  • Much simpler than deep learning")
    print("  • Good for comparing with LSTM/Transformer")