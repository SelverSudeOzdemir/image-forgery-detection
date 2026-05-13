import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict


class YOLODetection:
    """YOLO v8 ile tek goruntude manipulasyon tespiti"""

    def __init__(self):
        try:
            self.model = YOLO("yolov8n.pt", task="detect")
            print("YOLO v8 modeli yuklendi")
        except Exception as e:
            print(f"YOLO modeli yukleme hatasi: {e}")
            self.model = None

    def calculate_image_quality(self, image: np.ndarray) -> Dict:
        """Goruntu kalitesini analiz et"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        brightness = gray.mean()
        quality_score = (
            (laplacian_var / 1000.0) * 0.4 +
            (contrast / 100.0) * 0.4 +
            (brightness / 255.0) * 0.2
        ) * 100
        return {
            "sharpness": round(min(laplacian_var, 100), 2),
            "contrast": round(min(contrast, 100), 2),
            "brightness": round(brightness, 2),
            "quality_score": round(min(quality_score, 100), 2),
        }

    def detect_ela(self, img: np.ndarray, quality: int = 90) -> Dict:
        """Error Level Analysis (ELA) - JPEG sikistrima analizi"""
        try:
            # Gecici JPEG kaydet ve tekrar yukle
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            # Fark hesapla
            diff = cv2.absdiff(img, decoded)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # ELA istatistikleri
            ela_mean = float(gray_diff.mean())
            ela_std = float(gray_diff.std())
            ela_max = float(gray_diff.max())

            # Bolgesel analiz - goruntouyu bloklara bol
            h, w = gray_diff.shape
            block_size = 32
            block_means = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray_diff[y:y + block_size, x:x + block_size]
                    block_means.append(float(block.mean()))

            if block_means:
                block_std = float(np.std(block_means))
                block_max = float(np.max(block_means))
                block_mean_avg = float(np.mean(block_means))
            else:
                block_std = 0
                block_max = 0
                block_mean_avg = 0

            # Sahtecilik skoru - yuksek varyans = supheli
            manipulation_score = min((block_std * 10 + ela_std * 5), 100)

            return {
                "ela_mean": round(ela_mean, 3),
                "ela_std": round(ela_std, 3),
                "ela_max": round(ela_max, 3),
                "block_variance": round(block_std, 3),
                "manipulation_score": round(manipulation_score, 2),
                "is_suspicious": manipulation_score > 30
            }
        except Exception as e:
            return {
                "ela_mean": 0, "ela_std": 0, "ela_max": 0,
                "block_variance": 0, "manipulation_score": 0,
                "is_suspicious": False, "error": str(e)
            }

    def detect_noise_analysis(self, img: np.ndarray) -> Dict:
        """Gurultu analizi - farkli noise seviyeleri manipulasyonu gosterir"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Gurultu haritasi
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            noise = cv2.absdiff(gray, denoised)

            # Bolgesel gurultu analizi
            h, w = noise.shape
            block_size = 64
            noise_levels = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = noise[y:y + block_size, x:x + block_size]
                    noise_levels.append(float(block.std()))

            if noise_levels:
                noise_std = float(np.std(noise_levels))
                noise_mean = float(np.mean(noise_levels))
                noise_variation = noise_std / max(noise_mean, 0.001)
            else:
                noise_std = 0
                noise_mean = 0
                noise_variation = 0

            # Yuksek noise varyasyonu = farkli kaynaklardan birlestirilmis olabilir
            is_suspicious = noise_variation > 0.5

            return {
                "noise_mean": round(noise_mean, 3),
                "noise_std": round(noise_std, 3),
                "noise_variation": round(noise_variation, 3),
                "is_suspicious": is_suspicious
            }
        except Exception as e:
            return {"noise_mean": 0, "noise_std": 0, "noise_variation": 0, "is_suspicious": False, "error": str(e)}

    def detect_manipulations(self, img: np.ndarray) -> Dict:
        """Tek goruntude YOLO + ELA + Noise analizi"""
        try:
            # YOLO nesne tespiti
            detections = 0
            if self.model is not None:
                results = self.model(img, conf=0.25, verbose=False)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0

            # Kalite analizi
            quality = self.calculate_image_quality(img)

            # ELA analizi
            ela = self.detect_ela(img)

            # Noise analizi
            noise = self.detect_noise_analysis(img)

            # Genel karar
            suspicious_count = sum([
                ela["is_suspicious"],
                noise["is_suspicious"],
                quality["quality_score"] < 20
            ])

            confidence = (ela["manipulation_score"] * 0.5 + noise["noise_variation"] * 50 * 0.3 + (100 - quality["quality_score"]) * 0.2)
            confidence = min(confidence, 100)

            is_fake = suspicious_count >= 2 or ela["manipulation_score"] > 50

            return {
                "algorithm": "YOLO v8 + ELA",
                "confidence": round(confidence, 2),
                "is_fake": is_fake,
                "detection_count": detections,
                "quality": quality,
                "ela": ela,
                "noise": noise,
                "message": f"ELA Skor: {ela['manipulation_score']:.1f}, Noise Varyasyon: {noise['noise_variation']:.3f}"
            }
        except Exception as e:
            return {
                "algorithm": "YOLO v8 + ELA",
                "confidence": 0.0,
                "is_fake": False,
                "message": f"YOLO analiz hatasi: {str(e)}",
                "detection_count": 0,
                "quality": {},
                "ela": {},
                "noise": {}
            }
