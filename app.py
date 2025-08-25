import os
import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import tempfile
import gc
import math
import hashlib
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import warnings

# 修正 'ignore' 的引號
warnings.filterwarnings('ignore')

# Advanced ML libraries

import numpy as np
import joblib
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Enhanced ML toolkit

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin

# Configure advanced logging

logging.basicConfig(
    level=logging.INFO,
    # 修正 format 引號
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
# 修正 __name__ 語法
logger = logging.getLogger(__name__)

# 修正 Flask 和 CORS 的引號及 __name__ 語法
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"], supports_credentials=True)

# Advanced executor with priority queue

# 修正 thread_name_prefix 的引號
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="baccarat_ai_worker")

class AdvancedEnsembleModel(BaseEstimator, ClassifierMixin):
    """高級集成模型，結合多種算法的優勢""" # 修正 docstring 引號

    def __init__(self):
        # 初始化多個基礎模型
        self.models = {
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=80, max_depth=8, random_state=42,
                min_samples_split=3, min_samples_leaf=2
            ),
            'neural_net': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16), max_iter=500,
                random_state=42, alpha=0.01, learning_rate='adaptive'
            ),
            'logistic': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            )
        }

        self.model_weights = {}
        self.meta_classifier = LogisticRegression(random_state=42)
        self.is_fitted = False

    def fit(self, X, y):
        """訓練集成模型"""
        # 訓練各個基礎模型
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                # 計算模型權重（基於交叉驗證分數）
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                self.model_weights[name] = np.mean(scores)
                logger.info(f"Model {name} CV accuracy: {np.mean(scores):.3f}")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                self.model_weights[name] = 0.0

        # 正規化權重
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}

        # 訓練元分類器（堆疊學習）
        if len(X) >= 20:  # 確保有足夠數據進行堆疊
            try:
                meta_features = self._get_meta_features(X)
                self.meta_classifier.fit(meta_features, y)
            except Exception as e:
                logger.warning(f"Meta classifier training failed: {e}")

        self.is_fitted = True
        return self

    def _get_meta_features(self, X):
        """獲取元特徵（各模型的預測結果）"""
        meta_features = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba') and self.model_weights.get(name, 0) > 0:
                try:
                    probas = model.predict_proba(X)
                    meta_features.append(probas)
                except:
                    # 如果predict_proba失敗，使用predict
                    preds = model.predict(X).reshape(-1, 1)
                    # 假設有3個類別 (B, P, T)
                    meta_features.append(np.eye(3)[preds.flatten()]) 

        if meta_features:
            return np.hstack(meta_features)
        else:
            return X  # 如果無法獲取元特徵，使用原始特徵

    def predict_proba(self, X):
        """集成預測概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # 加權平均各模型的預測
        weighted_probas = None
        total_weight = 0

        for name, model in self.models.items():
            weight = self.model_weights.get(name, 0)
            if weight > 0:
                try:
                    probas = model.predict_proba(X)
                    if weighted_probas is None:
                        weighted_probas = probas * weight
                    else:
                        weighted_probas += probas * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")

        if weighted_probas is not None and total_weight > 0:
            weighted_probas /= total_weight

            # 如果元分類器可用，結合其預測
            try:
                meta_features = self._get_meta_features(X)
                meta_probas = self.meta_classifier.predict_proba(meta_features)
                # 組合加權平均和元分類器預測
                final_probas = 0.7 * weighted_probas + 0.3 * meta_probas
                return final_probas
            except:
                return weighted_probas

        # 如果所有模型都失敗，返回均勻分佈
        # 假設有3個類別 (B, P, T)
        return np.full((X.shape[0], 3), 1/3)

    def predict(self, X):
        """預測類別"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class UltimateModelManager:
    """終極模型管理器""" # 修正 docstring 引號

    def __init__(self):
        self._lock = threading.RLock()
        self.ensemble_model = None
        self.label_encoder = LabelEncoder()
        self.feature_scaler = RobustScaler()  # 對異常值更魯棒

        # 進階馬可夫鏈（多階）
        self.markov_chains = {
            1: defaultdict(lambda: defaultdict(int)),  # 1階
            2: defaultdict(lambda: defaultdict(int)),  # 2階  
            3: defaultdict(lambda: defaultdict(int)),  # 3階
            4: defaultdict(lambda: defaultdict(int))   # 4階
        }

        # 深度學習特徵
        self.sequence_memory = deque(maxlen=100)  # 序列記憶
        self.pattern_memory = defaultdict(list)   # 模式記憶
        self.adaptive_weights = defaultdict(float) # 自適應權重

        # 模型狀態
        self.is_initialized = False
        self.is_training = False
        self.last_training_time = 0
        self.model_performance_history = deque(maxlen=50)
        self.prediction_accuracy = 0.0

        # 在線學習緩衝區
        self.online_buffer_X = deque(maxlen=200)
        self.online_buffer_y = deque(maxlen=200)

        # 檔案路徑
        self.temp_dir = tempfile.gettempdir()
        self.MODEL_FILE = os.path.join(self.temp_dir, 'ultimate_baccarat_model.joblib')
        self.SCALER_FILE = os.path.join(self.temp_dir, 'ultimate_scaler.joblib')
        self.MEMORY_FILE = os.path.join(self.temp_dir, 'ultimate_memory.json')

        # 初始化標籤編碼器，適合百家樂的 B, P, T 三種結果
        self.label_encoder.fit(['B', 'P', 'T'])

    def update_online_buffer(self, features, label):
        """更新在線學習緩衝區"""
        with self._lock:
            self.online_buffer_X.append(features)
            self.online_buffer_y.append(label)

    def incremental_learning(self):
        """增量學習"""
        if len(self.online_buffer_X) < 10:  # 至少10個樣本
            logger.info("Not enough samples for incremental learning.")
            return False

        try:
            with self._lock:
                X_new = np.array(list(self.online_buffer_X))
                y_new = np.array(list(self.online_buffer_y))

                if self.ensemble_model is not None and self.is_initialized:
                    # 部分重訓練（僅針對神經網路）
                    if hasattr(self.ensemble_model, 'models'):
                        nn_model = self.ensemble_model.models.get('neural_net')
                        if nn_model is not None:
                            X_scaled = model_manager.feature_scaler.transform(X_new)
                            nn_model.partial_fit(X_scaled, y_new)
                            logger.info("Incremental learning completed")

                # 清除一半的緩衝區
                half_size = len(self.online_buffer_X) // 2
                for _ in range(half_size):
                    self.online_buffer_X.popleft()
                    self.online_buffer_y.popleft()

                return True

        except Exception as e:
            logger.error(f"Incremental learning failed: {e}")
            return False

    def save_memory_state(self):
        """保存記憶狀態"""
        try:
            memory_data = {
                'sequence_memory': list(self.sequence_memory),
                'pattern_memory': dict(self.pattern_memory),
                'adaptive_weights': dict(self.adaptive_weights),
                'performance_history': list(self.model_performance_history),
                'timestamp': time.time()
            }
            with open(self.MEMORY_FILE, 'w') as f:
                json.dump(memory_data, f, default=str)
            logger.info(f"Memory state saved successfully to {self.MEMORY_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save memory state: {e}")

    def load_memory_state(self):
        """載入記憶狀態"""
        try:
            if os.path.exists(self.MEMORY_FILE):
                with open(self.MEMORY_FILE, 'r') as f:
                    memory_data = json.load(f)

                self.sequence_memory = deque(memory_data.get('sequence_memory', []), maxlen=100)
                self.pattern_memory = defaultdict(list, memory_data.get('pattern_memory', {}))
                self.adaptive_weights = defaultdict(float, memory_data.get('adaptive_weights', {}))
                self.model_performance_history = deque(memory_data.get('performance_history', []), maxlen=50)

                logger.info(f"Memory state loaded successfully from {self.MEMORY_FILE}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load memory state: {e}")
        return False

# Global model manager instance
model_manager = UltimateModelManager()

# 頂級常數設定

LOOK_BACK = 12  # 增加序列長度
MARKOV_ORDERS = [1, 2, 3, 4]  # 多階馬可夫鏈
MIN_TRAINING_DATA = 50  # 提高最小訓練數據要求
MAX_HISTORY_SIZE = 10000  # 大幅增加歷史容量
LONG_RUN_THRESHOLD = 4
TREND_FOLLOW_START_THRESHOLD = 3
FEATURE_CACHE_SIZE = 2000  # 增加快取容量

# 超級遊戲歷史管理器

class UltimateGameHistory:
    def __init__(self, max_size=MAX_HISTORY_SIZE):
        # 修正 __init__ 語法
        self._history = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._max_size = max_size
        # 多層次快取 (暫時簡化為一個字典)
        self._cache = {} # 可以根據需要實現更複雜的 LRU 緩存

    def add_entry(self, entry: Dict):
        """添加新的遊戲紀錄"""
        with self._lock:
            self._history.append(entry)
            # 可以在這裡更新其他記憶體結構，例如 model_manager.sequence_memory

    def get_latest_entries(self, count: int) -> List[Dict]:
        """獲取最近的指定數量紀錄"""
        with self._lock:
            return list(self._history)[-count:]

    def get_all_entries(self) -> List[Dict]:
        """獲取所有遊戲紀錄"""
        with self._lock:
            return list(self._history)

    def clear(self):
        """清空所有遊戲紀錄"""
        with self._lock:
            self._history.clear()

# Global instance of game history
game_history = UltimateGameHistory()

# Helper function to generate dummy features for training/prediction
def generate_dummy_features(num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """生成虛擬特徵和標籤，用於模型訓練和預測的示例。
    假設有10個特徵，3個類別 (0, 1, 2 對應 'B', 'P', 'T')
    """
    X = np.random.rand(num_samples, 10) # 10個特徵
    y = np.random.randint(0, 3, num_samples) # 3個類別
    return X, y

# Flask Routes
@app.route('/train', methods=['POST'])
def train_model_route():
    """觸發模型訓練"""
    if model_manager.is_training:
        return jsonify({'status': 'training_in_progress', 'message': 'Model is already training.'}), 202

    data = request.get_json(silent=True)
    X_train, y_train_encoded = None, None

    if data and 'features' in data and 'labels' in data:
        try:
            X_train = np.array(data['features'])
            y_train_str = np.array(data['labels'])
            y_train_encoded = model_manager.label_encoder.transform(y_train_str)
            if X_train.shape[0] != y_train_encoded.shape[0]:
                raise ValueError("Features and labels must have the same number of samples.")
            if X_train.shape[0] < MIN_TRAINING_DATA:
                logger.warning(f"Provided data ({X_train.shape[0]} samples) is less than MIN_TRAINING_DATA ({MIN_TRAINING_DATA}).")
        except Exception as e:
            logger.error(f"Error processing provided training data: {e}")
            return jsonify({'status': 'error', 'message': f'Invalid training data: {str(e)}'}), 400
    else:
        # 如果沒有提供訓練數據，生成虛擬數據進行初始訓練
        logger.info("No training data provided or invalid format, using dummy data for initial training.")
        X_train, y_train_encoded = generate_dummy_features(MIN_TRAINING_DATA * 2) # 使用比最小值更多的數據以增加健壯性

    # 異步訓練
    def async_train_task(X, y):
        with model_manager._lock:
            model_manager.is_training = True
            try:
                logger.info(f"Starting model training with {X.shape[0]} samples.")
                # 特徵縮放
                X_scaled = model_manager.feature_scaler.fit_transform(X)
                
                model_manager.ensemble_model = AdvancedEnsembleModel()
                model_manager.ensemble_model.fit(X_scaled, y)
                model_manager.is_initialized = True
                model_manager.last_training_time = time.time()
                
                # 計算訓練集準確度
                if X_scaled.shape[0] > 0:
                    dummy_preds = model_manager.ensemble_model.predict(X_scaled)
                    dummy_accuracy = accuracy_score(y, dummy_preds)
                    model_manager.prediction_accuracy = dummy_accuracy
                    model_manager.model_performance_history.append(dummy_accuracy)
                    logger.info(f"Model training completed. Training Accuracy: {dummy_accuracy:.3f}")
                else:
                    model_manager.prediction_accuracy = 0.0
                    logger.warning("No data to evaluate model accuracy after training.")
                
                # 保存模型和scaler
                joblib.dump(model_manager.ensemble_model, model_manager.MODEL_FILE)
                joblib.dump(model_manager.feature_scaler, model_manager.SCALER_FILE)
                model_manager.save_memory_state()
                logger.info(f"Model and scaler saved to {model_manager.MODEL_FILE} and {model_manager.SCALER_FILE}")
            except Exception as e:
                logger.error(f"Error during async training: {e}")
            finally:
                model_manager.is_training = False
                logger.info("Model training task finished.")

    executor.submit(async_train_task, X_train, y_train_encoded)
    return jsonify({'status': 'training_started', 'message': 'Model training initiated in background.'}), 200


@app.route('/predict', methods=['POST'])
def predict_route():
    """進行模型預測"""
    if not model_manager.is_initialized or model_manager.is_training:
        status_msg = 'Model not initialized' if not model_manager.is_initialized else 'Model currently training'
        return jsonify({'status': 'error', 'message': f'{status_msg}. Please train the model first.'}), 400

    data = request.get_json(silent=True)
    if not data or 'features' not in data:
        return jsonify({'status': 'error', 'message': 'Missing "features" in request body.'}), 400

    try:
        features = np.array(data['features']).reshape(1, -1)
        
        # 檢查特徵數量是否符合訓練時的特徵數量
        if model_manager.feature_scaler.n_features_in_ is not None and \
           features.shape[1] != model_manager.feature_scaler.n_features_in_:
            return jsonify({
                'status': 'error',
                'message': f'Feature count mismatch. Expected {model_manager.feature_scaler.n_features_in_}, got {features.shape[1]}.'
            }), 400

        scaled_features = model_manager.feature_scaler.transform(features)
        
        probas = model_manager.ensemble_model.predict_proba(scaled_features)
        prediction_idx = np.argmax(probas, axis=1)[0]
        prediction_label = model_manager.label_encoder.inverse_transform([prediction_idx])[0]
        
        # 更新在線學習緩衝區
        if 'actual_label' in data:
            actual_label_str = data['actual_label']
            if actual_label_str in model_manager.label_encoder.classes_:
                actual_label_encoded = model_manager.label_encoder.transform([actual_label_str])[0]
                model_manager.update_online_buffer(features.flatten(), actual_label_encoded)
                # 如果緩衝區足夠大，觸發增量學習
                if len(model_manager.online_buffer_X) >= 50: # 示例閾值
                    executor.submit(model_manager.incremental_learning)
            else:
                logger.warning(f"Invalid actual_label '{actual_label_str}' provided. Skipping online buffer update.")

        return jsonify({
            'status': 'success',
            'prediction': prediction_label,
            'probabilities': probas[0].tolist(),
            'model_accuracy': f"{model_manager.prediction_accuracy:.3f}"
        }), 200
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """獲取模型和系統狀態"""
    return jsonify({
        'is_initialized': model_manager.is_initialized,
        'is_training': model_manager.is_training,
        'last_training_time': time.ctime(model_manager.last_training_time) if model_manager.last_training_time else 'N/A',
        'prediction_accuracy': f"{model_manager.prediction_accuracy:.3f}",
        'online_buffer_size': len(model_manager.online_buffer_X),
        'history_size': len(game_history.get_all_entries())
    }), 200

# Route for adding game history entries (can be used for collecting data for training)
@app.route('/history', methods=['POST'])
def add_history_entry():
    """向遊戲歷史記錄添加條目"""
    data = request.get_json(silent=True)
    if not data or 'entry' not in data:
        return jsonify({'status': 'error', 'message': 'Missing "entry" in request body.'}), 400
    
    game_history.add_entry(data['entry'])
    return jsonify({'status': 'success', 'message': 'Entry added to history.'}), 200

@app.route('/history', methods=['GET'])
def get_history():
    """獲取所有遊戲歷史記錄"""
    return jsonify({'status': 'success', 'history': game_history.get_all_entries()}), 200

if __name__ == '__main__':
    # 應用程式啟動時，先嘗試載入記憶狀態
    model_manager.load_memory_state()
    # 如果模型和scaler檔案存在，則載入它們
    try:
        if os.path.exists(model_manager.MODEL_FILE) and os.path.exists(model_manager.SCALER_FILE):
            model_manager.ensemble_model = joblib.load(model_manager.MODEL_FILE)
            model_manager.feature_scaler = joblib.load(model_manager.SCALER_FILE)
            model_manager.is_initialized = True
            logger.info("Existing model and scaler loaded successfully.")
            # 載入後嘗試計算一次準確度，如果歷史數據夠
            if len(model_manager.online_buffer_X) > 0:
                X_buff = np.array(list(model_manager.online_buffer_X))
                y_buff = np.array(list(model_manager.online_buffer_y))
                X_scaled_buff = model_manager.feature_scaler.transform(X_buff)
                if X_scaled_buff.shape[0] > 0:
                    preds_buff = model_manager.ensemble_model.predict(X_scaled_buff)
                    model_manager.prediction_accuracy = accuracy_score(y_buff, preds_buff)
                    logger.info(f"Loaded model accuracy on buffer data: {model_manager.prediction_accuracy:.3f}")
        else:
            logger.info("No existing model or scaler found. Model will be trained on first request or with dummy data.")
    except Exception as e:
        logger.error(f"Failed to load existing model/scaler: {e}")
        model_manager.is_initialized = False # 如果載入失敗，重置狀態

    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
