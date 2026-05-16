from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from datetime import datetime
from models.traditional import TraditionalDetection
from models.yolo_model import YOLODetection
from models.ai_models import AIDetection

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "static/uploads"
# GIF, TIFF, WEBP de destekleniyor (User Story-1)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

print("Traditional basliyor...")
traditional_detector = TraditionalDetection()
print("Traditional tamam")

print("YOLO basliyor...")
yolo_detector = YOLODetection()
print("YOLO tamam")

print("AI basliyor...")
ai_detector = AIDetection()
print("AI tamam")

print("Tum modeller yuklendi!")


def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_safe(filepath: str) -> np.ndarray:
    """
    GIF dahil tüm formatları güvenli yükler.
    OpenCV GIF okuyamaz; bu yüzden Pillow fallback kullanılır.
    """
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext == "gif":
        try:
            from PIL import Image
            pil_img = Image.open(filepath)
            pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            raise ValueError(f"GIF yuklenemedi: {e}")
    else:
        img = cv2.imread(filepath)
        if img is None:
            # OpenCV başarısız olursa Pillow dene
            try:
                from PIL import Image
                pil_img = Image.open(filepath).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e2:
                raise ValueError(f"Goruntu yuklenemedi: {e2}")
        return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "Dosya gerekli"}), 400
        f = request.files["file"]
        if f.filename == "" or not allowed_file(f.filename):
            return jsonify({"success": False, "message": "Desteklenmeyen format. PNG, JPG, GIF, BMP, TIFF, WEBP kabul edilir."}), 400
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = f.filename.rsplit(".", 1)[1].lower()
        fn = secure_filename(f"upload_{ts}.{ext}")
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], fn))
        return jsonify({"success": True, "message": "Yuklendi", "filename": fn}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "filename" not in data:
            return jsonify({"success": False, "message": "Dosya adi eksik"}), 400
        fp = os.path.join(app.config["UPLOAD_FOLDER"], data["filename"])
        if not os.path.exists(fp):
            return jsonify({"success": False, "message": "Dosya bulunamadi"}), 400

        img = load_image_safe(fp)

        print(f"Analiz: {data['filename']} | Boyut: {img.shape}")
        results = {}

        print("  -> Geleneksel (SIFT, SURF, AKAZE, ORB)...")
        results["traditional"] = traditional_detector.analyze_all_traditional(img)

        print("  -> YOLO + ELA...")
        results["yolo"] = yolo_detector.detect_manipulations(img)

        print("  -> AI (CNN + LSTM)...")
        results["ai"] = ai_detector.analyze_all_ai(img)

        # Final karar - majority voting
        af = [
            results["traditional"]["overall"]["is_fake"],
            results["yolo"]["is_fake"],
            results["ai"]["overall"]["is_fake"]
        ]
        fc = sum(af)
        ac = [
            results["traditional"]["overall"]["confidence"],
            results["yolo"]["confidence"],
            results["ai"]["overall"]["confidence"]
        ]
        results["final_decision"] = {
            "is_fake": fc >= 2,
            "confidence": round(sum(ac) / len(ac), 2),
            "agreement": f"{fc}/3 model sahte olduğunu söylüyor",
            "votes": {
                "traditional": af[0],
                "yolo": af[1],
                "ai": af[2]
            }
        }
        print(f"Sonuc: {'SAHTE' if fc >= 2 else 'ORIJINAL'} | Oylar: {fc}/3")
        return jsonify({"success": True, "message": "Tamam", "results": results}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    try:
        data = request.get_json()
        if "filename" in data:
            fp = os.path.join(app.config["UPLOAD_FOLDER"], data["filename"])
            if os.path.exists(fp):
                os.remove(fp)
        return jsonify({"success": True}), 200
    except:
        return jsonify({"success": False}), 500


if __name__ == "__main__":
    print("http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
