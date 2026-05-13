import numpy as np
import cv2
from typing import Dict
import os


class AIDetection:
    """CNN ve LSTM benzeri analiz ile tek goruntude sahtecilik tespiti"""

    def __init__(self, model_path: str = "models/cnn_lstm_model.h5"):
        self.model_path = model_path
        print("AI Detection modulu yuklendi")

    def detect_cnn(self, img: np.ndarray) -> Dict:
        """CNN benzeri ozellik cikarma ile sahtecilik tespiti"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # --- Kenar tutarsizligi analizi ---
            edges = cv2.Canny(gray, 50, 150)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)

            # Bolgesel kenar yogunlugu
            block_size = 64
            edge_densities = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = edges[y:y + block_size, x:x + block_size]
                    edge_densities.append(float(block.mean()))

            edge_std = float(np.std(edge_densities)) if edge_densities else 0
            edge_mean = float(np.mean(edge_densities)) if edge_densities else 0

            # --- Renk tutarsizligi analizi ---
            b, g, r = cv2.split(img)
            color_blocks = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    b_block = b[y:y + block_size, x:x + block_size].mean()
                    g_block = g[y:y + block_size, x:x + block_size].mean()
                    r_block = r[y:y + block_size, x:x + block_size].mean()
                    color_blocks.append([b_block, g_block, r_block])

            color_blocks = np.array(color_blocks)
            color_variance = float(np.mean(np.std(color_blocks, axis=0))) if len(color_blocks) > 0 else 0

            # --- Doku tutarsizligi ---
            texture_scores = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y + block_size, x:x + block_size]
                    lap = cv2.Laplacian(block, cv2.CV_64F).var()
                    texture_scores.append(float(lap))

            texture_std = float(np.std(texture_scores)) if texture_scores else 0
            texture_mean = float(np.mean(texture_scores)) if texture_scores else 0

            # Sahtecilik skoru
            edge_score = min(edge_std * 2, 40)
            color_score = min(color_variance / 2, 30)
            texture_score = min(texture_std / 50, 30)

            total_score = edge_score + color_score + texture_score
            confidence = min(total_score, 100)

            is_fake = confidence > 40

            return {
                "algorithm": "CNN",
                "confidence": round(confidence, 2),
                "is_fake": is_fake,
                "edge_inconsistency": round(edge_std, 3),
                "color_variance": round(color_variance, 3),
                "texture_inconsistency": round(texture_std, 3),
                "message": f"CNN: Kenar={edge_std:.2f}, Renk={color_variance:.2f}, Doku={texture_std:.2f}"
            }
        except Exception as e:
            return {
                "algorithm": "CNN",
                "confidence": 0.0,
                "is_fake": False,
                "message": f"CNN analiz hatasi: {str(e)}"
            }

    def detect_lstm(self, img: np.ndarray) -> Dict:
        """LSTM benzeri ardisik analiz ile sahtecilik tespiti"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # --- Histogram analizi ---
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histogram_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))

            # --- Satir bazli ardisik analiz (LSTM benzeri) ---
            row_means = [float(gray[y, :].mean()) for y in range(0, h, max(h // 100, 1))]
            col_means = [float(gray[:, x].mean()) for x in range(0, w, max(w // 100, 1))]

            # Ani degisimler = splice kenarlari
            row_diffs = np.diff(row_means)
            col_diffs = np.diff(col_means)

            row_jumps = int(np.sum(np.abs(row_diffs) > np.std(row_diffs) * 3))
            col_jumps = int(np.sum(np.abs(col_diffs) > np.std(col_diffs) * 3))

            # --- DCT (Discrete Cosine Transform) analizi ---
            gray_float = np.float32(gray)
            block_size = 8
            dct_variances = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray_float[y:y + block_size, x:x + block_size]
                    dct = cv2.dct(block)
                    dct_variances.append(float(np.var(dct)))

            dct_std = float(np.std(dct_variances)) if dct_variances else 0
            dct_mean = float(np.mean(dct_variances)) if dct_variances else 0

            # --- JPEG Ghost analizi ---
            ghost_scores = []
            for q in [60, 70, 80, 90]:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
                _, encoded = cv2.imencode(".jpg", img, encode_param)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                diff = cv2.absdiff(img, decoded)
                ghost_scores.append(float(diff.mean()))

            ghost_variation = float(np.std(ghost_scores))

            # Sahtecilik skoru
            jump_score = min((row_jumps + col_jumps) * 3, 35)
            dct_score = min(dct_std / 100, 35)
            ghost_score = min(ghost_variation * 5, 30)

            total_score = jump_score + dct_score + ghost_score
            confidence = min(total_score, 100)

            is_fake = confidence > 40

            return {
                "algorithm": "LSTM",
                "confidence": round(confidence, 2),
                "is_fake": is_fake,
                "row_jumps": row_jumps,
                "col_jumps": col_jumps,
                "dct_variance": round(dct_std, 3),
                "ghost_variation": round(ghost_variation, 3),
                "histogram_entropy": round(histogram_entropy, 3),
                "message": f"LSTM: Atlamalar={row_jumps + col_jumps}, DCT={dct_std:.2f}, Ghost={ghost_variation:.3f}"
            }
        except Exception as e:
            return {
                "algorithm": "LSTM",
                "confidence": 0.0,
                "is_fake": False,
                "message": f"LSTM analiz hatasi: {str(e)}"
            }

    def analyze_all_ai(self, img: np.ndarray) -> Dict:
        """Tum AI yontemlerini calistir"""
        cnn_result = self.detect_cnn(img)
        lstm_result = self.detect_lstm(img)

        results = {"cnn": cnn_result, "lstm": lstm_result}

        avg_confidence = (cnn_result.get("confidence", 0) + lstm_result.get("confidence", 0)) / 2
        fake_votes = sum(1 for r in [cnn_result, lstm_result] if r.get("is_fake", False))

        results["overall"] = {
            "confidence": round(avg_confidence, 2),
            "is_fake": fake_votes >= 1,
            "votes": fake_votes
        }

        return results
