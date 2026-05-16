import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict
import os


class YOLODetection:
    """YOLOv8 CASIA modeli + ELA + Noise + kalite analizi"""

    FAKE_LABELS = {
        "tp", "tampered", "fake", "forged", "spliced",
        "copymove", "copy_move", "manipulated", "sahte", "1",
    }
    REAL_LABELS = {
        "au", "authentic", "real", "original", "genuine", "orijinal", "0",
    }

    def __init__(self):
        try:
            model_path = "models/casia_model.pt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model bulunamadi: {model_path}")
            self.model = YOLO(model_path)
            if hasattr(self.model, 'names'):
                print(f"CASIA YOLO modeli yuklendi. Class'lar: {self.model.names}")
            else:
                print("CASIA YOLO modeli yuklendi (class bilgisi yok)")
        except Exception as e:
            print(f"YOLO modeli yukleme hatasi: {e}")
            self.model = None

    def _is_fake_label(self, label: str, class_id: int) -> bool:
        label_lower = label.lower().strip()
        if label_lower in self.FAKE_LABELS:
            return True
        if label_lower in self.REAL_LABELS:
            return False
        return class_id == 0

    def calculate_image_quality(self, image: np.ndarray) -> Dict:
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
            "sharpness": round(float(min(laplacian_var, 100)), 2),
            "contrast": round(float(min(contrast, 100)), 2),
            "brightness": round(float(brightness), 2),
            "quality_score": round(float(min(quality_score, 100)), 2),
        }

    def detect_ela(self, img: np.ndarray, quality: int = 90) -> Dict:
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            diff = cv2.absdiff(img, decoded)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ela_mean = float(gray_diff.mean())
            ela_std = float(gray_diff.std())
            ela_max = float(gray_diff.max())
            h, w = gray_diff.shape
            block_size = 32
            block_means = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray_diff[y:y + block_size, x:x + block_size]
                    block_means.append(float(block.mean()))
            block_std = float(np.std(block_means)) if block_means else 0.0
            manipulation_score = float(min((block_std * 10 + ela_std * 5), 100))
            return {
                "ela_mean": round(ela_mean, 3),
                "ela_std": round(ela_std, 3),
                "ela_max": round(ela_max, 3),
                "block_variance": round(block_std, 3),
                "manipulation_score": round(manipulation_score, 2),
                "is_suspicious": bool(manipulation_score > 30),
            }
        except Exception as e:
            return {"ela_mean": 0.0, "ela_std": 0.0, "ela_max": 0.0,
                    "block_variance": 0.0, "manipulation_score": 0.0,
                    "is_suspicious": False, "error": str(e)}

    def detect_noise_analysis(self, img: np.ndarray) -> Dict:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            noise = cv2.absdiff(gray, denoised)
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
                noise_variation = float(noise_std / max(noise_mean, 0.001))
            else:
                noise_std = noise_mean = noise_variation = 0.0
            return {
                "noise_mean": round(noise_mean, 3),
                "noise_std": round(noise_std, 3),
                "noise_variation": round(noise_variation, 3),
                "is_suspicious": bool(noise_variation > 0.5),
            }
        except Exception as e:
            return {"noise_mean": 0.0, "noise_std": 0.0, "noise_variation": 0.0,
                    "is_suspicious": False, "error": str(e)}

    def detect_yolo_classification(self, img: np.ndarray) -> Dict:
        try:
            if self.model is None:
                return {"label": "unknown", "confidence": 0.0,
                        "is_fake": False, "error": "Model yuklenemedi", "all_classes": {}}
            results = self.model(img, verbose=False)
            probs = results[0].probs
            class_id = int(probs.top1)
            confidence_cls = float(probs.top1conf)
            label = results[0].names[class_id]
            all_classes = {}
            if hasattr(probs, 'data'):
                for idx, prob in enumerate(probs.data.tolist()):
                    cls_name = results[0].names.get(idx, str(idx))
                    all_classes[cls_name] = round(float(prob * 100), 2)
            is_fake = bool(self._is_fake_label(label, class_id))
            return {
                "label": str(label),
                "class_id": int(class_id),
                "confidence": round(confidence_cls * 100, 2),
                "raw_confidence": float(confidence_cls),
                "is_fake": is_fake,
                "all_classes": all_classes,
            }
        except Exception as e:
            return {"label": "unknown", "class_id": -1,
                    "confidence": 0.0, "raw_confidence": 0.0,
                    "is_fake": False, "error": str(e), "all_classes": {}}

    def detect_manipulations(self, img: np.ndarray) -> Dict:
        try:
            yolo_result = self.detect_yolo_classification(img)
            quality = self.calculate_image_quality(img)
            ela = self.detect_ela(img)
            noise = self.detect_noise_analysis(img)
            is_yolo_fake = bool(yolo_result["is_fake"])
            confidence_cls = float(yolo_result["raw_confidence"])
            suspicious_count = int(sum([
                ela["is_suspicious"],
                noise["is_suspicious"],
                quality["quality_score"] < 20,
                is_yolo_fake,
            ]))
            confidence = float(
                ela["manipulation_score"] * 0.25 +
                noise["noise_variation"] * 50 * 0.15 +
                (100 - quality["quality_score"]) * 0.10 +
                confidence_cls * 100 * 0.50
            )
            confidence = float(min(confidence, 100))
            is_fake = bool((is_yolo_fake and confidence_cls > 0.55) or suspicious_count >= 3)
            return {
                "algorithm": "YOLOv8 CASIA + ELA + Noise",
                "confidence": round(confidence, 2),
                "is_fake": is_fake,
                "detection_count": int(1 if is_yolo_fake else 0),
                "predicted_label": str(yolo_result["label"]),
                "yolo_confidence": float(yolo_result["confidence"]),
                "yolo_class_id": int(yolo_result.get("class_id", -1)),
                "all_classes": yolo_result.get("all_classes", {}),
                "quality": quality,
                "ela": ela,
                "noise": noise,
                "message": (
                    f"YOLO: {yolo_result['label']} (class_id={yolo_result.get('class_id','?')}) "
                    f"({yolo_result['confidence']}%), "
                    f"ELA: {ela['manipulation_score']:.1f}, "
                    f"Noise: {noise['noise_variation']:.3f}"
                ),
            }
        except Exception as e:
            return {
                "algorithm": "YOLOv8 CASIA + ELA + Noise",
                "confidence": 0.0, "is_fake": False,
                "message": f"YOLO analiz hatasi: {str(e)}",
                "detection_count": 0, "predicted_label": "unknown",
                "yolo_confidence": 0.0, "quality": {}, "ela": {}, "noise": {},
                "all_classes": {},
            }