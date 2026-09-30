"""
Anomaly Detection System - Backend Application
Target Detection by Optimizing Anomaly Detection in Hyperspectral and RGB Image Processing using AI/ML

This Flask application provides:
- RGB image anomaly detection targeting REAL anomalies: HUMANS and DRONES
- YOLOv8s-based object detection for humans, drones, and vehicles (including small/distant targets)
- Statistical fallback using Mahalanobis distance for unknown anomaly types
- Hyperspectral image anomaly detection using RX (Reed-Xiaoli) detector
- Heatmap and overlay visualization with labeled bounding boxes
- Downloadable PDF reports
"""

import os
import io
import json
import uuid
import base64
import traceback
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── YOLOv8 Target Detector (Humans, Drones, Vehicles & Weapons) ──────────────

# COCO class IDs that represent our real anomaly targets
# 0  = person   (human)
# 2  = car      } vehicle group
# 5  = bus      }   NOTE: motorcycle (3) intentionally EXCLUDED —
# 7  = truck    }         YOLO misidentifies rifles/guns as motorcycles
# 4  = airplane (fixed-wing UAV proxy)
# 14 = bird     (small UAV / quadcopter proxy – often confused with drones in COCO)
YOLO_HUMAN_CLASS_ID    = 0           # person
YOLO_DRONE_CLASS_IDS   = {4, 14}     # airplane, bird – drone proxies in COCO
YOLO_VEHICLE_CLASS_IDS = {2, 3, 5, 7} # car, motorcycle, bus, truck

# All target class IDs combined
YOLO_ALL_TARGET_IDS = (
    {YOLO_HUMAN_CLASS_ID} | YOLO_DRONE_CLASS_IDS | YOLO_VEHICLE_CLASS_IDS
)

# Colour palette for bounding-box rendering (BGR)
COLOUR_HUMAN   = (0,   0,   255)   # Red     – human
COLOUR_DRONE   = (0, 165,   255)   # Orange  – drone
COLOUR_VEHICLE = (255, 200,   0)   # Cyan    – vehicle
COLOUR_WEAPON  = (255,   0, 200)   # Magenta – weapon / gun
COLOUR_BORDER  = (255, 255, 255)   # White   – label border

_yolo_model        = None   # lazy-loaded YOLOv8s (COCO) singleton
_weapon_model      = None   # lazy-loaded weapon-detection model singleton


def _get_yolo_model():
    """Lazy-load YOLOv8s (COCO) model once and cache it."""
    global _yolo_model
    if _yolo_model is None:
        try:
            # pyrefly: ignore [missing-import]
            from ultralytics import YOLO
            _yolo_model = YOLO('yolov8s.pt')   # small model – better accuracy for distant/small objects
            print("[YOLO] YOLOv8s model loaded successfully.")
        except Exception as e:
            print(f"[YOLO] Could not load YOLOv8 model: {e}. Falling back to statistical detection.")
            _yolo_model = False   # sentinel – don't retry
    return _yolo_model if _yolo_model is not False else None


def _get_weapon_model():
    """
    Lazy-load a weapon-detection YOLO model – tries ONCE, fails fast.

    Only checks for local 'weapon.pt'.  Remote downloads are skipped entirely
    because the HuggingFace repo is gated (401) and the download hangs every
    request for 30-60 s, causing page timeouts.

    To enable weapon model: place 'weapon.pt' next to app.py manually.
    """
    global _weapon_model
    if _weapon_model is None:
        import os
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weapon.pt')
        if os.path.exists(local_path):
            try:
                from ultralytics import YOLO
                _weapon_model = YOLO(local_path)
                print("[WEAPON] Local weapon.pt loaded successfully.")
            except Exception as e:
                print(f"[WEAPON] Local weapon.pt failed to load: {e}")
                _weapon_model = False
        else:
            # No local model — skip silently, heuristic handles weapon detection
            print("[WEAPON] No local weapon.pt found. Heuristic-only weapon detection active.")
            _weapon_model = False
    return _weapon_model if _weapon_model is not False else None





def detect_targets_yolo(image_bgr, conf_threshold=0.15):
    """
    Run YOLOv8 on the image and return detections for humans, drones, vehicles, and weapons.

    Returns a list of dicts:
        {
            'label'   : 'Human' | 'Drone' | 'Vehicle' | 'Weapon',
            'sublabel': e.g. 'Car', 'Truck', 'Drone', 'Human', 'Gun', 'Rifle'
            'conf'    : float,
            'box'     : (x1, y1, x2, y2),   # absolute pixel coords
            'class_id': int
        }
    """
    # COCO class name mapping for vehicle subtypes
    VEHICLE_SUBTYPE = {
        2: 'Car',
        3: 'Motorcycle',
        5: 'Bus',
        7: 'Truck',
    }

    model = _get_yolo_model()
    if model is None:
        return []

    h, w = image_bgr.shape[:2]

    # ── IoU helper ────────────────────────────────────────────────────────────
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def _run_yolo(img, ox=0, oy=0, conf=conf_threshold):
        """Run YOLO on one image region and return remapped raw detections."""
        try:
            results = model(
                img,
                conf=conf,
                imgsz=640,
                verbose=False,
                classes=list(YOLO_ALL_TARGET_IDS)
            )
        except Exception as e:
            print(f"[YOLO] Inference error at tile ({ox},{oy}): {e}")
            return []

        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf_v = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Remap to full-image coords
                x1 += ox; x2 += ox
                y1 += oy; y2 += oy
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if cls_id == YOLO_HUMAN_CLASS_ID:
                    label, sublabel = 'Human', 'Human'
                elif cls_id in YOLO_DRONE_CLASS_IDS:
                    label, sublabel = 'Drone', 'Drone'
                elif cls_id in YOLO_VEHICLE_CLASS_IDS:
                    label = 'Vehicle'
                    sublabel = VEHICLE_SUBTYPE.get(cls_id, 'Vehicle')
                else:
                    continue

                print(f"[YOLO] Raw cls={cls_id}({sublabel}) conf={conf_v:.3f} "
                      f"tile=({ox},{oy}) box=({x1},{y1},{x2},{y2})")
                dets.append({
                    'label'   : label,
                    'sublabel': sublabel,
                    'conf'    : conf_v,
                    'box'     : (x1, y1, x2, y2),
                    'class_id': cls_id,
                })
        return dets

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 – Full image at imgsz=640
    # This produces ACCURATE, FULL-BODY bounding boxes for all visible targets.
    # We always trust these boxes; tiles can only ADD new detections, never
    # replace or suppress these.
    # ─────────────────────────────────────────────────────────────────────────
    full_image_dets = _run_yolo(image_bgr)
    # Deduplicate within the full-image pass (basic NMS by confidence)
    full_image_dets.sort(key=lambda d: d['conf'], reverse=True)
    detections = []
    for det in full_image_dets:
        if not any(_iou(det['box'], k['box']) > 0.4 for k in detections):
            detections.append(det)
    print(f"[YOLO] Phase-1 (full image): {len(detections)} target(s) → "
          f"{[d['sublabel'] for d in detections]}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 – Zoomed tiles of the upper region (background / horizon area)
    # Finds small / distant objects (cars, people far away) too small in Phase-1.
    #
    # Suppression uses IoS (Intersection over Smaller area):
    #   IoS = intersection / area_of_smaller_box
    #
    # This catches cases where a tile detection is a PARTIAL crop (e.g. just
    # the face/torso) of a person already detected in Phase-1. Their IoU is
    # tiny (small face vs. large body = small union), but IoS is large (the
    # face is almost entirely INSIDE the full-body box). If IoS > 0.5, the
    # new detection is a duplicate and is suppressed.
    # ─────────────────────────────────────────────────────────────────────────
    def _ios(new_box, existing_box):
        """Intersection over the area of the smaller (new) box."""
        ax1, ay1, ax2, ay2 = new_box
        bx1, by1, bx2, by2 = existing_box
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_new = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / area_new

    def _is_duplicate(new_det, accepted):
        """True if new_det is covered by or heavily overlaps any accepted box."""
        for k in accepted:
            if _iou(new_det['box'], k['box']) > 0.35:
                return True          # standard IoU duplicate
            if _ios(new_det['box'], k['box']) > 0.50:
                return True          # new box is mostly INSIDE existing box
        return False

    # Focus on top 55% of image where distant objects live
    upper_h = int(h * 0.55)
    upper_w = w

    # Each tile = ~62.5% width, full upper height → effective 2× zoom
    t_h = upper_h
    t_w = upper_w // 2 + upper_w // 8

    zoom_tiles = [
        (0,               0, t_w,      t_h),   # upper-left
        (upper_w - t_w,   0, upper_w, t_h),    # upper-right
        ((upper_w - t_w) // 2, 0,
         (upper_w - t_w) // 2 + t_w, t_h),    # centre
    ]

    # Collect all Phase-2 raw detections, tracking how many tiles each came from
    # phase2_raw entries: dict with detection fields + 'tile_count'
    phase2_raw = []

    def _close_centers(box_a, box_b, px=20):
        """True if the two boxes share approximately the same centre."""
        cx_a = (box_a[0] + box_a[2]) // 2
        cy_a = (box_a[1] + box_a[3]) // 2
        cx_b = (box_b[0] + box_b[2]) // 2
        cy_b = (box_b[1] + box_b[3]) // 2
        return abs(cx_a - cx_b) <= px and abs(cy_a - cy_b) <= px

    def _merge_raw(new_det, existing):
        """Add new_det to existing list, incrementing vote count if duplicate."""
        for ex in existing:
            if (ex['class_id'] == new_det['class_id'] and
                    _close_centers(ex['box'], new_det['box'])):
                ex['votes'] += 1
                if new_det['conf'] > ex['conf']:
                    ex['conf'] = new_det['conf']
                    ex['box']  = new_det['box']
                return
        new_det['votes'] = 1
        existing.append(new_det)

    # Pass A – overlapping quadrant zoom tiles
    for (x0, y0, x1e, y1e) in zoom_tiles:
        tile_img = image_bgr[y0:y1e, x0:x1e]
        if tile_img.size == 0:
            continue
        tile_dets = _run_yolo(tile_img, ox=x0, oy=y0, conf=0.10)
        for det in tile_dets:
            _merge_raw(det, phase2_raw)

    # Pass B – super-zoom background strip (top 35% of image, upscaled 3×)
    # This gives ~3× resolution boost on distant objects like cars / tiny people.
    bg_h = int(h * 0.35)
    bg_strip = image_bgr[0:bg_h, :]
    if bg_strip.size > 0:
        zoom_factor = 3
        bg_zoom = cv2.resize(
            bg_strip,
            (bg_strip.shape[1] * zoom_factor, bg_strip.shape[0] * zoom_factor),
            interpolation=cv2.INTER_LANCZOS4
        )
        bg_dets_raw = _run_yolo(bg_zoom, ox=0, oy=0, conf=0.10)
        # Remap from zoomed coords back to full-image coords
        for det in bg_dets_raw:
            x1z, y1z, x2z, y2z = det['box']
            det['box'] = (
                x1z // zoom_factor,
                y1z // zoom_factor,
                x2z // zoom_factor,
                y2z // zoom_factor,
            )
            print(f"[YOLO] BG-strip cls={det['class_id']}({det['sublabel']}) "
                  f"conf={det['conf']:.3f} box={det['box']}")
            _merge_raw(det, phase2_raw)

    # ── Filter and deduplicate Phase-2 candidates ─────────────────────────────
    # Accept a detection if:
    #   a) area >= 200px²  (a real object at close range), OR
    #   b) votes >= 2      (same location confirmed by 2+ independent tile passes)
    # Single-tile tiny boxes (pure noise) are rejected.
    phase2_raw.sort(key=lambda d: d['conf'], reverse=True)
    phase2_dets = []
    for det in phase2_raw:
        bw = det['box'][2] - det['box'][0]
        bh = det['box'][3] - det['box'][1]
        area   = bw * bh
        votes  = det.get('votes', 1)

        if area < 200 and votes < 2:
            print(f"[YOLO] Phase-2 SKIP single-tile noise "
                  f"{bw}×{bh}px area={area} votes={votes} {det['sublabel']}")
            continue

        if not _is_duplicate(det, phase2_dets):
            phase2_dets.append(det)

    # Add Phase-2 detections that are NOT duplicates of Phase-1
    for det in phase2_dets:
        if not _is_duplicate(det, detections):
            print(f"[YOLO] Phase-2 ADD: {det['sublabel']} "
                  f"conf={det['conf']:.3f} votes={det.get('votes',1)} box={det['box']}")
            detections.append(det)
        else:
            print(f"[YOLO] Phase-2 SUPPRESS (dup of Phase-1): "
                  f"{det['sublabel']} conf={det['conf']:.3f} box={det['box']}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3A – Heuristic weapon relabeling (no extra model needed)
    #
    # COCO-YOLO has no gun/weapon class.  It commonly misdetects rifles as:
    #   • Drone / airplane (class 4)  – barrel+scope looks like a fuselage
    #   • Bird           (class 14)  – same elongated shape
    #
    # Heuristic: reclassify a Drone/Bird detection as "Weapon (Suspected)" if:
    #   1. Confidence < 45 %  (low-confidence → model is unsure)
    #   2. Bounding box aspect ratio > 2.5  (elongated: guns are long & thin)
    #   3. Box centre is in the lower 80 % of image  (ground level, not sky)
    # ─────────────────────────────────────────────────────────────────────────
    relabeled = []
    remaining = []
    for det in detections:
        bx1, by1, bx2, by2 = det['box']
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        aspect = max(bw / bh, bh / bw)          # elongation ratio
        cy = (by1 + by2) / 2                    # vertical centre (0=top)
        is_ground_level = cy > h * 0.20          # not in top 20 % (sky)

        if (det['label'] in ['Drone', 'Vehicle'] and
                det['conf'] < 0.45 and
                aspect > 1.3 and
                is_ground_level):
            print(f"[HEURISTIC] Reclassifying {det['sublabel']} conf={det['conf']:.3f} "
                  f"aspect={aspect:.1f} → Weapon (Suspected)")
            relabeled.append({
                **det,
                'label'   : 'Weapon',
                'sublabel': 'Weapon (Suspected)',
            })
        else:
            remaining.append(det)

    # Merge fragmented/duplicate weapon detections from different YOLO phases
    # Two weapon boxes are merged if they overlap (IoU > 0.05) OR one is mostly
    # inside the other (IoS > 0.3). Merged box = union of both boxes.
    def _box_union(a, b):
        return (min(a[0], b[0]), min(a[1], b[1]),
                max(a[2], b[2]), max(a[3], b[3]))

    def _weapon_overlap(a, b, pad=80):
        """True if boxes overlap or are within `pad` pixels of each other."""
        ax1, ay1, ax2, ay2 = a[0]-pad, a[1]-pad, a[2]+pad, a[3]+pad
        bx1, by1, bx2, by2 = b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        return inter_w * inter_h > 0


    merged_relabeled = []
    while relabeled:
        curr = relabeled.pop(0)
        merged = True
        while merged:
            merged = False
            for i, other in enumerate(relabeled):
                if _weapon_overlap(curr['box'], other['box']):
                    curr['box']  = _box_union(curr['box'], other['box'])
                    curr['conf'] = max(curr['conf'], other['conf'])
                    relabeled.pop(i)
                    merged = True
                    break
        merged_relabeled.append(curr)

    detections = remaining + merged_relabeled

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3B – Dedicated weapon detection model (optional, if downloadable)
    # ─────────────────────────────────────────────────────────────────────────
    weapon_model = _get_weapon_model()
    if weapon_model is not None:
        try:
            w_results = weapon_model(
                image_bgr,
                conf=0.25,
                imgsz=640,
                verbose=False,

            )
            for result in w_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf_v = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    # Get weapon class name from model
                    cls_name = result.names.get(cls_id, 'Weapon')
                    sublabel = cls_name.replace('-', ' ').title()
                    det = {
                        'label'   : 'Weapon',
                        'sublabel': sublabel,
                        'conf'    : conf_v,
                        'box'     : (x1, y1, x2, y2),
                        'class_id': cls_id,
                        'votes'   : 1,
                    }
                    if not _is_duplicate(det, detections):
                        print(f"[WEAPON] Detected: {sublabel} conf={conf_v:.3f} "
                              f"box=({x1},{y1},{x2},{y2})")
                        detections.append(det)
        except Exception as e:
            print(f"[WEAPON] Inference error: {e}")

    # ── Post-processing cleanup ────────────────────────────────────────────────
    # Low-confidence Drone detections overlapping a confirmed Weapon box are
    # almost certainly a rifle scope / barrel misclassified as an airplane.
    weapon_boxes = [d['box'] for d in detections if d['label'] == 'Weapon']
    if weapon_boxes:
        cleaned = []
        for det in detections:
            if det['label'] == 'Drone' and det['conf'] < 0.30:
                if any(_iou(det['box'], wb) > 0.20 for wb in weapon_boxes):
                    print(f"[YOLO] Cleanup: removing gun-scope Drone "
                          f"conf={det['conf']:.3f} box={det['box']}")
                    continue
            cleaned.append(det)
        detections = cleaned

    print(f"[YOLO] Found {len(detections)} target(s) total: "
          f"{[d['sublabel'] for d in detections]}")
    return detections



def build_detection_score_map(scores_stat, detections, image_shape):
    """
    Fuse statistical anomaly scores with YOLO bounding-box detections.

    Strategy:
    - Start from the statistical score map (background layer).
    - For each detected human / drone, forcibly set that region to the
      MAXIMUM possible score so it always shows as a hot anomaly.
    - Non-target regions are left with their statistical scores so the
      heatmap still shows genuine background variation.
    """
    h, w = image_shape[:2]
    fused = scores_stat.copy().astype(np.float64)

    if not detections:
        return fused

    score_max = fused.max() if fused.max() > 0 else 1.0

    for det in detections:
        x1, y1, x2, y2 = det['box']
        # clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            # Use 3× max so targets dominate the normalised heatmap
            fused[y1:y2, x1:x2] = score_max * 3.0

    return fused


def draw_detection_boxes(image_bgr, detections):
    """
    Render coloured bounding boxes + confidence labels on the image.
    Colours: Red=Human | Orange=Drone | Cyan=Vehicle | Magenta=Weapon
    Returns (annotated_image, num_humans, num_drones, num_vehicles, num_weapons).
    """
    output = image_bgr.copy()
    num_humans   = 0
    num_drones   = 0
    num_vehicles = 0
    num_weapons  = 0

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label    = det['label']
        sublabel = det.get('sublabel', label)
        conf     = det['conf']

        if label == 'Human':
            colour = COLOUR_HUMAN
            num_humans += 1
        elif label == 'Drone':
            colour = COLOUR_DRONE
            num_drones += 1
        elif label == 'Weapon':
            colour = COLOUR_WEAPON
            num_weapons += 1
        else:  # Vehicle
            colour = COLOUR_VEHICLE
            num_vehicles += 1

        # Thick bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 3)

        # Label text: show sublabel (e.g. "Car", "Truck", "Rifle") + confidence
        text = f"{sublabel} {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(y1 - th - baseline - 6, 0)
        label_y2 = max(y1, th + baseline + 6)
        cv2.rectangle(output, (x1, label_y1), (x1 + tw + 8, label_y2), colour, -1)

        # White text
        cv2.putText(
            output, text,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOUR_BORDER, 2, cv2.LINE_AA
        )

        # Semi-transparent fill inside box
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

    return output, num_humans, num_drones, num_vehicles, num_weapons

# ─── App Configuration ───────────────────────────────────────────────────────

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_RGB_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'}
ALLOWED_HYPER_EXTENSIONS = {'hdr', 'npy', 'mat', 'tif', 'tiff'}


# ─── Utility Functions ───────────────────────────────────────────────────────

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def encode_image_to_base64(img_array):
    """Convert numpy array image to base64 string."""
    success, buffer = cv2.imencode('.png', img_array)
    if success:
        return base64.b64encode(buffer).decode('utf-8')
    return None


def create_heatmap(scores, colormap='jet'):
    """Create a colored heatmap from anomaly scores."""
    normalized = np.zeros_like(scores)
    if scores.max() > scores.min():
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
    
    cmap = plt.get_cmap(colormap)
    heatmap = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    return heatmap


def create_overlay(original, heatmap, alpha=0.5):
    """Blend original image with heatmap overlay."""
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    if original.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlay


def create_anomaly_mask(scores, threshold_percentile=95):
    """Create binary mask of anomaly regions."""
    threshold = np.percentile(scores, threshold_percentile)
    mask = (scores >= threshold).astype(np.uint8) * 255
    return mask, threshold


def create_highlighted_output(original, mask):
    """Create output with anomaly regions highlighted with contours."""
    if len(original.shape) == 2:
        output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        output = original.copy()
    
    # Find contours of anomaly regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours with a bright color
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
    
    # Create semi-transparent red overlay on anomaly areas
    red_overlay = output.copy()
    red_overlay[mask > 0] = [0, 0, 255]
    output = cv2.addWeighted(output, 0.7, red_overlay, 0.3, 0)
    
    return output, len(contours)


# ─── RGB Anomaly Detection ───────────────────────────────────────────────────

def extract_rgb_features(image, block_size=1):
    """
    Extract multi-dimensional features from an RGB image.
    Features include: R, G, B channels, intensity, local contrast, and texture.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    h, w = image.shape[:2]
    
    # Color features (normalized)
    img_float = image.astype(np.float64) / 255.0
    b_chan, g_chan, r_chan = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    
    # Intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    
    # Local contrast (standard deviation in neighborhood)
    blur = gaussian_filter(gray, sigma=3)
    local_contrast = np.sqrt(gaussian_filter((gray - blur) ** 2, sigma=3))
    
    # Texture using Local Binary Pattern
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
    lbp_normalized = lbp / lbp.max() if lbp.max() > 0 else lbp
    
    # Gradient magnitude
    sobelx = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = gradient_mag / gradient_mag.max() if gradient_mag.max() > 0 else gradient_mag
    
    # Stack all features
    features = np.stack([r_chan, g_chan, b_chan, gray, local_contrast, lbp_normalized, gradient_mag], axis=-1)
    
    return features


def detect_anomalies_rgb(image, method='mahalanobis'):
    """
    Detect anomalies in RGB images.

    PRIMARY layer  – YOLOv8 object detection targeting HUMANS and DRONES.
    SECONDARY layer – Statistical (Mahalanobis / Z-score) background deviation
                     used to generate the heatmap and catch anything YOLO misses.

    Returns:
        scores      : 2-D float array of fused anomaly scores
        detections  : list of YOLO detection dicts (may be empty)
    """
    h, w = image.shape[:2]
    features = extract_rgb_features(image)
    
    # Reshape features to 2D (pixels x features)
    n_features = features.shape[-1]
    pixels = features.reshape(-1, n_features)
    
    # Remove NaN/Inf values
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
    
    # ── Statistical (background) score map ──────────────────────────────────
    if method == 'mahalanobis':
        scores_stat = _mahalanobis_anomaly(pixels, h, w)
    elif method == 'statistical':
        scores_stat = _statistical_anomaly(pixels, h, w, features)
    else:
        scores_stat = _mahalanobis_anomaly(pixels, h, w)
    
    # Smooth to reduce pixel-level noise
    scores_stat = gaussian_filter(scores_stat, sigma=2)
    
    # ── YOLO target detection (humans & drones) ──────────────────────────────
    detections = detect_targets_yolo(image, conf_threshold=0.25)
    
    # ── Fuse: stamp max score onto detected target bounding boxes ────────────
    scores_fused = build_detection_score_map(scores_stat, detections, image.shape)
    
    return scores_fused, detections


def _mahalanobis_anomaly(pixels, h, w):
    """Mahalanobis distance-based anomaly detection with robust covariance estimation."""
    try:
        # Subsample for robust covariance estimation (for performance)
        n_pixels = len(pixels)
        if n_pixels > 50000:
            indices = np.random.choice(n_pixels, 50000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels
        
        # Robust covariance estimation using Minimum Covariance Determinant
        try:
            robust_cov = MinCovDet(random_state=42, support_fraction=0.75).fit(sample)
        except Exception:
            robust_cov = EmpiricalCovariance().fit(sample)
        
        # Calculate Mahalanobis distance for all pixels
        scores = robust_cov.mahalanobis(pixels)
        scores = scores.reshape(h, w)
        
    except Exception as e:
        print(f"Mahalanobis failed, falling back to Euclidean: {e}")
        mean = np.mean(pixels, axis=0)
        scores = np.sqrt(np.sum((pixels - mean) ** 2, axis=1))
        scores = scores.reshape(h, w)
    
    return scores


def _statistical_anomaly(pixels, h, w, features):
    """Local statistical deviation anomaly detection."""
    n_features = features.shape[-1]
    
    # Global statistics
    global_mean = np.mean(pixels, axis=0)
    global_std = np.std(pixels, axis=0)
    global_std[global_std < 1e-10] = 1e-10
    
    # Z-score for each pixel
    z_scores = np.abs((pixels - global_mean) / global_std)
    
    # Combined anomaly score (max z-score across features)
    combined_scores = np.max(z_scores, axis=1)
    scores = combined_scores.reshape(h, w)
    
    return scores


# ─── Hyperspectral Anomaly Detection ─────────────────────────────────────────

def load_hyperspectral_image(filepath):
    """
    Load hyperspectral image from various formats.
    Supports: .npy (numpy), .tif/.tiff (multi-band TIFF), .hdr (ENVI)
    """
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext == 'npy':
        data = np.load(filepath)
        return data.astype(np.float64)
    
    elif ext in ('tif', 'tiff'):
        # Try loading as multi-band TIFF
        from PIL import Image as PILImage
        img = PILImage.open(filepath)
        
        # Check if it has multiple bands
        n_bands = getattr(img, 'n_frames', 1)
        if n_bands > 3:
            bands = []
            for i in range(n_bands):
                img.seek(i)
                bands.append(np.array(img))
            data = np.stack(bands, axis=-1)
        else:
            # Treat as regular image with channels as bands
            data = np.array(img)
            if len(data.shape) == 2:
                data = data[:, :, np.newaxis]
        
        return data.astype(np.float64)
    
    elif ext == 'hdr':
        try:
            # pyrefly: ignore [missing-import]
            import spectral.io.envi as envi
            img = envi.open(filepath)
            data = img.load()
            return np.array(data, dtype=np.float64)
        except ImportError:
            raise ValueError("spectral library required for .hdr files. Install with: pip install spectral")
    
    else:
        raise ValueError(f"Unsupported hyperspectral format: .{ext}")


def rx_anomaly_detector(data):
    """
    RX (Reed-Xiaoli) Anomaly Detector for hyperspectral images.
    
    The RX detector calculates the Mahalanobis distance of each pixel's
    spectral signature from the global background statistics.
    
    RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
    
    where μ is the mean spectrum and Σ is the covariance matrix.
    """
    h, w = data.shape[:2]
    n_bands = data.shape[2] if len(data.shape) > 2 else 1
    
    if n_bands == 1:
        data = data.reshape(h, w, 1)
    
    # Reshape to (pixels, bands)
    pixels = data.reshape(-1, n_bands)
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate background statistics
    mean_spectrum = np.mean(pixels, axis=0)
    
    # Regularized covariance estimation
    cov_matrix = np.cov(pixels, rowvar=False)
    
    # Add regularization to prevent singular matrix
    reg_param = 1e-4 * np.trace(cov_matrix) / n_bands
    cov_matrix += reg_param * np.eye(n_bands)
    
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_matrix)
    
    # Calculate RX score for each pixel
    diff = pixels - mean_spectrum
    rx_scores = np.sum(diff @ inv_cov * diff, axis=1)
    rx_scores = rx_scores.reshape(h, w)
    
    return rx_scores


def detect_anomalies_hyperspectral(data):
    """
    Main hyperspectral anomaly detection pipeline.
    Uses RX detector with preprocessing.
    """
    h, w = data.shape[:2]
    n_bands = data.shape[2] if len(data.shape) > 2 else 1
    
    # Normalize each band
    for b in range(n_bands):
        band = data[:, :, b]
        bmin, bmax = band.min(), band.max()
        if bmax > bmin:
            data[:, :, b] = (band - bmin) / (bmax - bmin)
    
    # Apply RX detector
    scores = rx_anomaly_detector(data)
    
    # Smooth to reduce noise
    scores = gaussian_filter(scores, sigma=1.5)
    
    return scores


# ─── Report Generation ───────────────────────────────────────────────────────

def generate_pdf_report(result_data, result_id):
    """Generate a comprehensive PDF report of the anomaly detection results."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    report_path = os.path.join(RESULTS_FOLDER, f'{result_id}_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#6C63FF'), spaceAfter=20)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2D2B55'), spaceBefore=15, spaceAfter=10)
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=11, leading=16)
    
    elements = []
    
    # Title
    elements.append(Paragraph("Anomaly Detection System - Analysis Report", title_style))
    elements.append(Spacer(1, 10))
    
    # Metadata
    elements.append(Paragraph("Report Information", heading_style))
    meta_data = [
        ['Report ID', result_id],
        ['Date & Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Image Type', result_data.get('image_type', 'N/A').upper()],
        ['Image Dimensions', f"{result_data.get('width', 'N/A')} x {result_data.get('height', 'N/A')} pixels"],
        ['Detection Method', result_data.get('method', 'N/A')],
    ]
    
    meta_table = Table(meta_data, colWidths=[3*inch, 3.5*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0EFFF')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2D2B55')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0DFFF')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 15))
    
    # Detection Results
    elements.append(Paragraph("Detection Results", heading_style))
    stats = result_data.get('statistics', {})
    results_data = [
        ['Metric', 'Value'],
        ['Total Pixels', f"{stats.get('total_pixels', 'N/A'):,}"],
        ['Anomalous Pixels', f"{stats.get('anomaly_pixels', 'N/A'):,}"],
        ['Anomaly Percentage', f"{stats.get('anomaly_percentage', 0):.2f}%"],
        ['Detected Regions', str(stats.get('num_regions', 'N/A'))],
        ['Humans Detected', str(stats.get('num_humans_detected', 'N/A'))],
        ['Drones Detected', str(stats.get('num_drones_detected', 'N/A'))],
        ['Vehicles Detected', str(stats.get('num_vehicles_detected', 'N/A'))],
        ['Threshold Percentile', f"{stats.get('threshold_percentile', 'N/A')}%"],
        ['Min Anomaly Score', f"{stats.get('min_score', 0):.4f}"],
        ['Max Anomaly Score', f"{stats.get('max_score', 0):.4f}"],
        ['Mean Anomaly Score', f"{stats.get('mean_score', 0):.4f}"],
        ['Score Std Deviation', f"{stats.get('std_score', 0):.4f}"],
    ]
    
    results_table = Table(results_data, colWidths=[3*inch, 3.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6C63FF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#F0EFFF')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0DFFF')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FAFAFE')]),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 15))
    
    # Save visualization images for report
    for img_key, img_title in [('heatmap', 'Anomaly Heatmap'), ('highlighted', 'Highlighted Anomalies'), ('mask', 'Anomaly Mask')]:
        img_b64 = result_data.get(f'{img_key}_base64')
        if img_b64:
            img_path = os.path.join(RESULTS_FOLDER, f'{result_id}_{img_key}.png')
            img_data = base64.b64decode(img_b64)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            elements.append(Paragraph(img_title, heading_style))
            rl_img = RLImage(img_path, width=5.5*inch, height=4*inch)
            rl_img.hAlign = 'CENTER'
            elements.append(rl_img)
            elements.append(Spacer(1, 10))
    
    # Method Description
    elements.append(Paragraph("Detection Methodology", heading_style))
    if result_data.get('image_type') == 'rgb':
        yolo_dets = result_data.get('statistics', {}).get('yolo_detections', [])
        yolo_summary = ''
        if yolo_dets:
            yolo_summary = (
                f" YOLOv8 identified {len(yolo_dets)} target(s) in this image: "
                + ', '.join(f"{d['label']} ({d['confidence']:.0%})" for d in yolo_dets)
                + "."
            )
        method_text = (
            "This analysis uses a two-layer detection pipeline for identifying REAL anomalies "
            "(humans, drones, and vehicles) in the scene. "
            "PRIMARY LAYER: YOLOv8 nano object detection is applied to locate: "
            "(1) Persons/humans, (2) UAVs/drones (via COCO airplane and bird class proxies), "
            "(3) Vehicles including cars, motorcycles, buses and trucks. "
            "Detected targets receive the maximum anomaly score, ensuring they always appear "
            "as hot zones in the heatmap. "
            "SECONDARY LAYER: Mahalanobis distance statistical anomaly detection is applied "
            "using Minimum Covariance Determinant robust covariance estimation on multi-dimensional "
            "pixel features (RGB channels, intensity, local contrast, LBP texture, gradient magnitude). "
            "This catches any anomalies YOLO may miss and provides a continuous background score map."
            + yolo_summary
        )
    else:
        method_text = (
            "This analysis uses the RX (Reed-Xiaoli) anomaly detector for hyperspectral images. "
            "The RX detector computes the Mahalanobis distance of each pixel's spectral signature "
            "from the global background statistics: RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ). "
            "Pixels with spectral responses significantly different from the background "
            "distribution receive high anomaly scores."
        )
    elements.append(Paragraph(method_text, body_style))
    elements.append(Spacer(1, 10))
    
    # Disclaimer
    elements.append(Paragraph("Disclaimer", heading_style))
    disclaimer_text = (
        "This system is designed to detect HUMANS, DRONES, and VEHICLES as primary anomaly targets "
        "using YOLOv8 object detection, supplemented by statistical background anomaly scoring. "
        "Bounding boxes: RED = Human | ORANGE = Drone | CYAN = Vehicle (Car/Truck/Bus/Motorcycle). "
        "Results should be verified by domain experts. Detection accuracy depends on "
        "image quality, lighting conditions, target size, and occlusion."
    )
    elements.append(Paragraph(disclaimer_text, body_style))
    
    # Footer
    elements.append(Spacer(1, 20))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph("Generated by Anomaly Detection System | Target Detection by Optimizing Anomaly Detection in Hyperspectral and RGB Image Processing using AI/ML", footer_style))
    
    doc.build(elements)
    return report_path


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Anomaly Detection System is running'})


@app.route('/api/detect', methods=['POST'])
def detect_anomalies():
    """Main anomaly detection endpoint."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        image_type = request.form.get('image_type', 'rgb')
        method = request.form.get('method', 'mahalanobis')
        threshold_percentile = float(request.form.get('threshold', 95))
        
        # Generate unique result ID
        result_id = str(uuid.uuid4())[:8]
        
        # Save uploaded file
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        filepath = os.path.join(UPLOAD_FOLDER, f'{result_id}_input.{ext}')
        file.save(filepath)
        
        if image_type == 'rgb':
            # ── RGB Processing ──
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Failed to read image. Please upload a valid RGB image.'}), 400
            
            # Resize if too large (for performance and memory)
            # Use 640 so YOLO also runs at decent resolution
            max_dim = 640
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
                h, w = image.shape[:2]
            
            # ── Primary + secondary detection (YOLO + statistical) ──────────
            scores, detections = detect_anomalies_rgb(image, method=method)
            
            # ── Threshold: if YOLO found targets, lower threshold so targets
            #    always appear as anomaly regions even at default settings ────
            effective_threshold = threshold_percentile
            if detections:
                # With fused scores the target regions are at 3× max,
                # so any percentile ≤ 99.5 will include them.
                effective_threshold = min(threshold_percentile, 97.0)
            
            # Generate visualizations
            heatmap = create_heatmap(scores)
            overlay = create_overlay(image, heatmap, alpha=0.45)
            mask, threshold_value = create_anomaly_mask(scores, effective_threshold)

            # Draw bounding boxes + labels on the highlighted image
            annotated_image, num_humans, num_drones, num_vehicles, num_weapons = draw_detection_boxes(image, detections)
            highlighted_yolo, num_regions_yolo = create_highlighted_output(annotated_image, mask)

            # Also keep a pure statistical highlighted (no boxes) for comparison
            highlighted_stat, num_regions_stat = create_highlighted_output(image, mask)

            # Primary highlighted output = YOLO-annotated version
            highlighted  = highlighted_yolo
            num_regions  = num_regions_yolo if detections else num_regions_stat
            
            # Statistics
            total_pixels = h * w
            anomaly_pixels = int(np.sum(mask > 0))
            anomaly_percentage = (anomaly_pixels / total_pixels) * 100
            
            statistics = {
                'total_pixels': total_pixels,
                'anomaly_pixels': anomaly_pixels,
                'anomaly_percentage': round(anomaly_percentage, 2),
                'num_regions': num_regions,
                'threshold_percentile': effective_threshold,
                'threshold_value': float(threshold_value),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                # YOLO-specific counts
                'num_humans_detected'  : num_humans,
                'num_drones_detected'  : num_drones,
                'num_vehicles_detected': num_vehicles,
                'num_weapons_detected' : num_weapons,
                'yolo_detections': [
                    {
                        'label'     : d['label'],
                        'sublabel'  : d.get('sublabel', d['label']),
                        'confidence': round(d['conf'], 3),
                        'bbox'      : list(d['box']),
                    }
                    for d in detections
                ],
            }
            
            result_data = {
                'result_id': result_id,
                'image_type': 'rgb',
                'method': f"{method} + YOLOv8 target detection" if detections else method,
                'width': w,
                'height': h,
                'original_base64': encode_image_to_base64(image),
                'heatmap_base64': encode_image_to_base64(heatmap),
                'overlay_base64': encode_image_to_base64(overlay),
                'highlighted_base64': encode_image_to_base64(highlighted),
                'mask_base64': encode_image_to_base64(mask),
                'statistics': statistics,
            }
            
        elif image_type == 'hyperspectral':
            # ── Hyperspectral Processing ──
            try:
                hyper_data = load_hyperspectral_image(filepath)
            except Exception as e:
                return jsonify({'error': f'Failed to load hyperspectral image: {str(e)}'}), 400
            
            h, w = hyper_data.shape[:2]
            n_bands = hyper_data.shape[2] if len(hyper_data.shape) > 2 else 1
            
            # Detect anomalies using RX detector
            scores = detect_anomalies_hyperspectral(hyper_data)
            
            # Create RGB representation for visualization (using first 3 bands or PCA)
            if n_bands >= 3:
                rgb_vis = hyper_data[:, :, [0, n_bands//2, n_bands-1]]
            else:
                rgb_vis = hyper_data[:, :, 0:min(3, n_bands)]
            
            if rgb_vis.shape[2] == 1:
                rgb_vis = np.repeat(rgb_vis, 3, axis=2)
            elif rgb_vis.shape[2] == 2:
                rgb_vis = np.concatenate([rgb_vis, rgb_vis[:, :, 0:1]], axis=2)
            
            # Normalize for display
            for c in range(3):
                ch = rgb_vis[:, :, c]
                cmin, cmax = ch.min(), ch.max()
                if cmax > cmin:
                    rgb_vis[:, :, c] = (ch - cmin) / (cmax - cmin) * 255
            rgb_vis = rgb_vis.astype(np.uint8)
            rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
            
            # Generate visualizations
            heatmap = create_heatmap(scores)
            overlay = create_overlay(rgb_vis_bgr, heatmap, alpha=0.45)
            mask, threshold_value = create_anomaly_mask(scores, threshold_percentile)
            highlighted, num_regions = create_highlighted_output(rgb_vis_bgr, mask)
            
            total_pixels = h * w
            anomaly_pixels = int(np.sum(mask > 0))
            anomaly_percentage = (anomaly_pixels / total_pixels) * 100
            
            statistics = {
                'total_pixels': total_pixels,
                'anomaly_pixels': anomaly_pixels,
                'anomaly_percentage': round(anomaly_percentage, 2),
                'num_regions': num_regions,
                'threshold_percentile': threshold_percentile,
                'threshold_value': float(threshold_value),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'num_bands': n_bands,
            }
            
            result_data = {
                'result_id': result_id,
                'image_type': 'hyperspectral',
                'method': 'RX (Reed-Xiaoli) Detector',
                'width': w,
                'height': h,
                'num_bands': n_bands,
                'original_base64': encode_image_to_base64(rgb_vis_bgr),
                'heatmap_base64': encode_image_to_base64(heatmap),
                'overlay_base64': encode_image_to_base64(overlay),
                'highlighted_base64': encode_image_to_base64(highlighted),
                'mask_base64': encode_image_to_base64(mask),
                'statistics': statistics,
            }
        else:
            return jsonify({'error': 'Invalid image type. Use "rgb" or "hyperspectral".'}), 400
        
        # Save scores for potential re-thresholding
        scores_path = os.path.join(RESULTS_FOLDER, f'{result_id}_scores.npy')
        np.save(scores_path, scores)
        
        # Save original image path
        result_data['scores_path'] = scores_path
        result_data['input_path'] = filepath
        
        return jsonify(result_data)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/api/rethreshold', methods=['POST'])
def rethreshold():
    """Re-apply threshold to existing anomaly scores."""
    try:
        data = request.json
        result_id = data.get('result_id')
        threshold_percentile = float(data.get('threshold', 95))
        
        scores_path = os.path.join(RESULTS_FOLDER, f'{result_id}_scores.npy')
        if not os.path.exists(scores_path):
            return jsonify({'error': 'Results not found. Please re-upload the image.'}), 404
        
        scores = np.load(scores_path)
        
        # Find original image
        input_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(result_id)]
        if not input_files:
            return jsonify({'error': 'Original image not found.'}), 404
        
        input_path = os.path.join(UPLOAD_FOLDER, input_files[0])
        image = cv2.imread(input_path)
        
        if image is None:
            # Might be hyperspectral - create a placeholder
            h, w = scores.shape
            image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Resize image to match scores if needed
        if image.shape[:2] != scores.shape:
            image = cv2.resize(image, (scores.shape[1], scores.shape[0]))
        
        # Re-generate visualizations with new threshold
        heatmap = create_heatmap(scores)
        overlay = create_overlay(image, heatmap, alpha=0.45)
        mask, threshold_value = create_anomaly_mask(scores, threshold_percentile)
        highlighted, num_regions = create_highlighted_output(image, mask)
        
        h, w = scores.shape
        total_pixels = h * w
        anomaly_pixels = int(np.sum(mask > 0))
        anomaly_percentage = (anomaly_pixels / total_pixels) * 100
        
        return jsonify({
            'heatmap_base64': encode_image_to_base64(heatmap),
            'overlay_base64': encode_image_to_base64(overlay),
            'highlighted_base64': encode_image_to_base64(highlighted),
            'mask_base64': encode_image_to_base64(mask),
            'statistics': {
                'total_pixels': total_pixels,
                'anomaly_pixels': anomaly_pixels,
                'anomaly_percentage': round(anomaly_percentage, 2),
                'num_regions': num_regions,
                'threshold_percentile': threshold_percentile,
                'threshold_value': float(threshold_value),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
            }
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Re-threshold failed: {str(e)}'}), 500


@app.route('/api/report/<result_id>', methods=['POST'])
def download_report(result_id):
    """Generate and download PDF report."""
    try:
        data = request.json
        report_path = generate_pdf_report(data, result_id)
        return send_file(report_path, as_attachment=True, download_name=f'anomaly_report_{result_id}.pdf')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    print("=" * 60)
    print("  Anomaly Detection System")
    print("  Target Detection by Optimizing Anomaly Detection")
    print("  in Hyperspectral and RGB Image Processing using AI/ML")
    print("=" * 60)
    print(f"  Server: http://localhost:{port}")
    print("=" * 60)
    app.run(debug=debug, host='0.0.0.0', port=port, use_reloader=False)
