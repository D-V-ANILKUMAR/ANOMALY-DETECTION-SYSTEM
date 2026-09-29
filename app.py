"""
Anomaly Detection System - Backend Application
Target Detection by Optimizing Anomaly Detection in Hyperspectral and RGB Image Processing using AI/ML

This Flask application provides:
- RGB image anomaly detection using statistical methods (Mahalanobis distance, Local Outlier Factor)
- Hyperspectral image anomaly detection using RX (Reed-Xiaoli) detector
- Heatmap and overlay visualization
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
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

ALLOWED_RGB_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
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
    Detect anomalies in RGB images using statistical methods.
    
    Methods:
    - mahalanobis: Mahalanobis distance from global mean (robust covariance)
    - statistical: Local statistical deviation method
    """
    h, w = image.shape[:2]
    features = extract_rgb_features(image)
    
    # Reshape features to 2D (pixels x features)
    n_features = features.shape[-1]
    pixels = features.reshape(-1, n_features)
    
    # Remove NaN/Inf values
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
    
    if method == 'mahalanobis':
        scores = _mahalanobis_anomaly(pixels, h, w)
    elif method == 'statistical':
        scores = _statistical_anomaly(pixels, h, w, features)
    else:
        scores = _mahalanobis_anomaly(pixels, h, w)
    
    # Smooth scores to reduce noise
    scores = gaussian_filter(scores, sigma=2)
    
    return scores


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
        method_text = (
            "This analysis uses statistical anomaly detection on RGB images. "
            "Multi-dimensional features (color channels, intensity, local contrast, "
            "texture via Local Binary Patterns, and gradient magnitude) are extracted from each pixel. "
            "The Mahalanobis distance from the robust global statistics is computed using "
            "Minimum Covariance Determinant estimation, identifying pixels that deviate "
            "significantly from the normal background distribution."
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
        "This system detects unusual visual/spectral patterns in the uploaded image. "
        "It does not identify or classify what the anomaly is (e.g., car, building, defect). "
        "Results should be verified by domain experts. Detection accuracy depends on "
        "image quality, lighting conditions, and the nature of anomalies present."
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
            
            # Resize if too large (for performance)
            max_dim = 1024
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
                h, w = image.shape[:2]
            
            # Detect anomalies
            scores = detect_anomalies_rgb(image, method=method)
            
            # Generate visualizations
            heatmap = create_heatmap(scores)
            overlay = create_overlay(image, heatmap, alpha=0.45)
            mask, threshold_value = create_anomaly_mask(scores, threshold_percentile)
            highlighted, num_regions = create_highlighted_output(image, mask)
            
            # Statistics
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
            }
            
            result_data = {
                'result_id': result_id,
                'image_type': 'rgb',
                'method': method,
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
    app.run(debug=debug, host='0.0.0.0', port=port)
