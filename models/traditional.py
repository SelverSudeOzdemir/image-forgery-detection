import cv2
import numpy as np
from typing import Dict, List

class TraditionalDetection:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.akaze = cv2.AKAZE_create()
        self.orb = cv2.ORB_create(nfeatures=5000)
        try:
            self.surf = cv2.xfeatures2d.SURF_create(400)
            print("SURF yuklendi")
        except:
            self.surf = None
            print("SURF yuklenemedi - patentli algoritma")

    def _find_copy_move_matches(self, keypoints, descriptors, algorithm_name, 
                                 min_distance=40.0, norm_type=cv2.NORM_L2, max_distance=80):
        try:
            if descriptors is None or len(keypoints) < 10:
                return {"algorithm": algorithm_name, "confidence": 5.0, "is_fake": False, 
                        "match_count": 0, "suspicious_regions": 0, 
                        "message": f"{algorithm_name}: Yetersiz ozellik ({len(keypoints) if keypoints else 0})"}
            bf = cv2.BFMatcher(norm_type)
            matches = bf.knnMatch(descriptors, descriptors, k=5)
            suspicious = []
            for mg in matches:
                for m in mg:
                    if m.queryIdx == m.trainIdx:
                        continue
                    pt1 = keypoints[m.queryIdx].pt
                    pt2 = keypoints[m.trainIdx].pt
                    sd = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                    if m.distance < max_distance and sd > min_distance:
                        suspicious.append({"pt1": pt1, "pt2": pt2, "dd": m.distance, "sd": sd})
            clusters = self._cluster(suspicious)
            ratio = len(suspicious) / max(len(keypoints), 1)
            conf = min(ratio * 500 + len(clusters) * 15, 100)
            fake = len(suspicious) > 50 and len(clusters) >= 3
            return {"algorithm": algorithm_name, "confidence": round(conf, 2), "is_fake": fake, 
                    "match_count": len(suspicious), "suspicious_regions": len(clusters), 
                    "total_keypoints": len(keypoints), 
                    "message": f"{algorithm_name}: {len(suspicious)} eslesmesi, {len(clusters)} bolge"}
        except Exception as e:
            return {"algorithm": algorithm_name, "confidence": 0.0, "is_fake": False, 
                    "match_count": 0, "suspicious_regions": 0, 
                    "message": f"{algorithm_name} hatasi: {str(e)}"}

    def _cluster(self, matches, cd=50.0):
        if not matches:
            return []
        pts = [(m["pt1"], m["pt2"]) for m in matches]
        clusters, used = [], set()
        for i, (a1, b1) in enumerate(pts):
            if i in used:
                continue
            c = [i]
            used.add(i)
            for j, (a2, b2) in enumerate(pts):
                if j in used:
                    continue
                if np.sqrt((a1[0]-a2[0])**2+(a1[1]-a2[1])**2) < cd and \
                   np.sqrt((b1[0]-b2[0])**2+(b1[1]-b2[1])**2) < cd:
                    c.append(j)
                    used.add(j)
            if len(c) >= 3:
                clusters.append(c)
        return clusters

    def detect_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return self._find_copy_move_matches(kp, des, "SIFT", 40.0, cv2.NORM_L2, max_distance=80)

    def detect_surf(self, img):
        if self.surf is None:
            return {"algorithm": "SURF", "confidence": 0.0, "is_fake": False, 
                    "match_count": 0, "suspicious_regions": 0, 
                    "message": "SURF kullanillamiyor (patentli)"}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return self._find_copy_move_matches(kp, des, "SURF", 40.0, cv2.NORM_L2, max_distance=80)

    def detect_akaze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.akaze.detectAndCompute(gray, None)
        return self._find_copy_move_matches(kp, des, "AKAZE", 40.0, cv2.NORM_HAMMING, max_distance=50)

    def detect_orb(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return self._find_copy_move_matches(kp, des, "ORB", 40.0, cv2.NORM_HAMMING, max_distance=30)

    def analyze_all_traditional(self, img):
        results = {
            "sift": self.detect_sift(img),
            "surf": self.detect_surf(img),
            "akaze": self.detect_akaze(img),
            "orb": self.detect_orb(img)
        }
        confs = [r["confidence"] for r in results.values()]
        fv = sum(1 for r in results.values() if r["is_fake"])
        results["overall"] = {"confidence": round(sum(confs)/len(confs), 2), 
                              "is_fake": fv >= 2, "votes": fv}
        return results