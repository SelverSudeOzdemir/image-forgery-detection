import numpy as np
import cv2
from typing import Dict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AIDetection:
    def __init__(self):
        self.cnn_model = None
        self.lstm_model = None
        self._load_models()

    def _load_models(self):
        try:
            import tensorflow as tf
            cnn_path = "models/cnn_model.h5"
            lstm_path = "models/lstm_model.h5"
            if os.path.exists(cnn_path):
                self.cnn_model = tf.keras.models.load_model(cnn_path)
                print("CNN modeli yuklendi")
            else:
                print("CNN model dosyasi bulunamadi")
            if os.path.exists(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                print("LSTM modeli yuklendi")
            else:
                print("LSTM model dosyasi bulunamadi")
        except Exception as e:
            print(f"Model yukleme hatasi: {e}")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, (128, 128))
        return img_resized.astype(np.float32) / 255.0

    def detect_cnn(self, img: np.ndarray) -> Dict:
        try:
            if self.cnn_model is None:
                return self._heuristic_cnn(img)

            processed = self._preprocess(img)
            input_data = np.expand_dims(processed, axis=0)
            prediction = float(self.cnn_model.predict(input_data, verbose=0)[0][0])
            is_fake = bool(prediction > 0.5)
            confidence = float(prediction * 100 if is_fake else (1 - prediction) * 100)

            return {
                "algorithm": "CNN",
                "confidence": round(confidence, 2),
                "is_fake": is_fake,
                "raw_score": round(prediction, 4),
                "message": f"CNN: {'Sahte' if is_fake else 'Orijinal'} ({confidence:.1f}%)"
            }
        except Exception as e:
            return self._heuristic_cnn(img)

    def detect_lstm(self, img: np.ndarray) -> Dict:
        try:
            if self.lstm_model is None:
                return self._heuristic_lstm(img)

            processed = self._preprocess(img)
            lstm_input = processed.reshape(1, 128, 128 * 3)
            prediction = float(self.lstm_model.predict(lstm_input, verbose=0)[0][0])
            is_fake = bool(prediction > 0.5)
            confidence = float(prediction * 100 if is_fake else (1 - prediction) * 100)

            return {
                "algorithm": "LSTM",
                "confidence": round(confidence, 2),
                "is_fake": is_fake,
                "raw_score": round(prediction, 4),
                "message": f"LSTM: {'Sahte' if is_fake else 'Orijinal'} ({confidence:.1f}%)"
            }
        except Exception as e:
            return self._heuristic_lstm(img)

    def _heuristic_cnn(self, img: np.ndarray) -> Dict:
        """Model yoksa heuristic fallback"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        block_size = 64
        edge_densities = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = edges[y:y+block_size, x:x+block_size]
                edge_densities.append(float(block.mean()))
        edge_std = float(np.std(edge_densities)) if edge_densities else 0
        b, g, r = cv2.split(img)
        color_blocks = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                color_blocks.append([
                    b[y:y+block_size, x:x+block_size].mean(),
                    g[y:y+block_size, x:x+block_size].mean(),
                    r[y:y+block_size, x:x+block_size].mean()
                ])
        color_variance = float(np.mean(np.std(np.array(color_blocks), axis=0))) if color_blocks else 0
        texture_scores = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                texture_scores.append(float(cv2.Laplacian(block, cv2.CV_64F).var()))
        texture_std = float(np.std(texture_scores)) if texture_scores else 0
        confidence = float(min(edge_std * 0.8 + color_variance / 4 + texture_std / 100, 100))
        return {
            "algorithm": "CNN",
            "confidence": round(confidence, 2),
            "is_fake": bool(confidence > 65),
            "edge_inconsistency": round(edge_std, 3),
            "color_variance": round(color_variance, 3),
            "message": f"CNN (heuristic): Kenar={edge_std:.2f}"
        }

    def _heuristic_lstm(self, img: np.ndarray) -> Dict:
        """Model yoksa heuristic fallback"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        row_means = [float(gray[y, :].mean()) for y in range(0, h, max(h//100, 1))]
        col_means = [float(gray[:, x].mean()) for x in range(0, w, max(w//100, 1))]
        row_jumps = int(np.sum(np.abs(np.diff(row_means)) > np.std(np.diff(row_means)) * 3))
        col_jumps = int(np.sum(np.abs(np.diff(col_means)) > np.std(np.diff(col_means)) * 3))
        gray_float = np.float32(gray)
        dct_variances = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                dct = cv2.dct(gray_float[y:y+8, x:x+8])
                dct_variances.append(float(np.var(dct)))
        dct_std = float(np.std(dct_variances)) if dct_variances else 0
        ghost_scores = []
        for q in [60, 70, 80, 90]:
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            ghost_scores.append(float(cv2.absdiff(img, dec).mean()))
        ghost_variation = float(np.std(ghost_scores))
        confidence = float(min((row_jumps + col_jumps) * 3 + dct_std / 100 + ghost_variation * 5, 100))
        return {
            "algorithm": "LSTM",
            "confidence": round(confidence, 2),
            "is_fake": bool(confidence > 65),
            "row_jumps": row_jumps,
            "dct_variance": round(dct_std, 3),
            "ghost_variation": round(ghost_variation, 3),
            "message": f"LSTM (heuristic): Atlamalar={row_jumps+col_jumps}"
        }

    def analyze_all_ai(self, img: np.ndarray) -> Dict:
        cnn_result = self.detect_cnn(img)
        lstm_result = self.detect_lstm(img)
        results = {"cnn": cnn_result, "lstm": lstm_result}
        avg_confidence = float((cnn_result.get("confidence", 0) + lstm_result.get("confidence", 0)) / 2)
        fake_votes = int(sum(1 for r in [cnn_result, lstm_result] if r.get("is_fake", False)))
        results["overall"] = {
            "confidence": round(avg_confidence, 2),
            "is_fake": bool(fake_votes >= 2),
            "votes": fake_votes
        }
        return results