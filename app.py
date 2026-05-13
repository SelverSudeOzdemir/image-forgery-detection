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
ALLOWED_EXTENSIONS = {"png","jpg","jpeg","gif","bmp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50*1024*1024

print("Modeller yukleniyor...")
traditional_detector = TraditionalDetection()
yolo_detector = YOLODetection()
ai_detector = AIDetection()
print("Tum modeller yuklendi!")

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"success":False,"message":"Dosya gerekli"}),400
        f = request.files["file"]
        if f.filename == "" or not allowed_file(f.filename):
            return jsonify({"success":False,"message":"Gecersiz dosya"}),400
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = f.filename.rsplit(".",1)[1].lower()
        fn = secure_filename(f"upload_{ts}.{ext}")
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], fn))
        return jsonify({"success":True,"message":"Yuklendi","filename":fn}),200
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}),500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "filename" not in data:
            return jsonify({"success":False,"message":"Dosya adi eksik"}),400
        fp = os.path.join(app.config["UPLOAD_FOLDER"], data["filename"])
        if not os.path.exists(fp):
            return jsonify({"success":False,"message":"Dosya bulunamadi"}),400
        img = cv2.imread(fp)
        if img is None:
            return jsonify({"success":False,"message":"Goruntu yuklenemedi"}),400
        print(f"Analiz: {data['filename']}")
        results = {}
        print("  -> Geleneksel (SIFT, SURF, AKAZE, ORB)...")
        results["traditional"] = traditional_detector.analyze_all_traditional(img)
        print("  -> YOLO + ELA...")
        results["yolo"] = yolo_detector.detect_manipulations(img)
        print("  -> AI (CNN + LSTM)...")
        results["ai"] = ai_detector.analyze_all_ai(img)
        af = [results["traditional"]["overall"]["is_fake"], results["yolo"]["is_fake"], results["ai"]["overall"]["is_fake"]]
        fc = sum(af)
        ac = [results["traditional"]["overall"]["confidence"], results["yolo"]["confidence"], results["ai"]["overall"]["confidence"]]
        results["final_decision"] = {"is_fake": fc>=2, "confidence": round(sum(ac)/len(ac),2), "agreement": f"{fc}/3 model sahte olduğunu söylüyor"}
        print(f"Sonuc: {'SAHTE' if fc>=2 else 'ORIJINAL'}")
        return jsonify({"success":True,"message":"Tamam","results":results}),200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"message":str(e)}),500

@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    try:
        data = request.get_json()
        if "filename" in data:
            fp = os.path.join(app.config["UPLOAD_FOLDER"], data["filename"])
            if os.path.exists(fp): os.remove(fp)
        return jsonify({"success":True}),200
    except:
        return jsonify({"success":False}),500

if __name__ == "__main__":
    print("http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
