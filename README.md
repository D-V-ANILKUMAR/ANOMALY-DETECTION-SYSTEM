# Anomaly Detection System

## Target Detection by Optimizing Anomaly Detection in Hyperspectral and RGB Image Processing using AI/ML

A web-based image anomaly detection application that analyzes uploaded RGB and hyperspectral images to identify unusual patterns and regions using statistical methods — **no external training dataset required**.

---

### Features

- **RGB Image Analysis** — Mahalanobis distance and statistical deviation methods  
- **Hyperspectral Image Analysis** — RX (Reed-Xiaoli) anomaly detector  
- **Interactive Threshold Adjustment** — Fine-tune detection sensitivity in real-time  
- **Multiple Visualizations** — Original, heatmap, overlay, highlighted contours, and binary mask  
- **Downloadable PDF Reports** — Comprehensive analysis report with statistics and visualizations  
- **No Training Required** — Per-image statistical analysis on uploaded images  

---

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| Image Processing | OpenCV, NumPy, SciPy, scikit-learn, scikit-image |
| Hyperspectral | spectral (SPy), RX Detector |
| Report Generation | ReportLab (PDF) |
| Frontend | HTML5, CSS3, JavaScript (Vanilla) |

---

### Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py
```

The server will start at **http://localhost:5000**

---

### Usage

1. Open `http://localhost:5000` in your browser  
2. Upload an RGB image (PNG, JPG, TIFF) or hyperspectral image (NPY, HDR)  
3. Select image type and detection method  
4. Adjust anomaly threshold (default: 95th percentile)  
5. Click "Run Anomaly Detection"  
6. View results: heatmap, overlay, highlighted regions, mask  
7. Download PDF report  

---

### Detection Methods

#### RGB Images
- **Mahalanobis Distance (Robust)** — Extracts multi-dimensional features (color, intensity, contrast, texture, gradient) and computes Mahalanobis distance using Minimum Covariance Determinant estimation  
- **Statistical Deviation** — Z-score analysis across feature dimensions to detect extreme outliers  

#### Hyperspectral Images  
- **RX (Reed-Xiaoli) Detector** — Computes spectral Mahalanobis distance: `RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)`. Standard algorithm for hyperspectral anomaly detection  

---

### Project Structure

```
ANOMALY DETECTION SYSTEM/
├── app.py              # Flask backend with detection algorithms
├── requirements.txt    # Python dependencies
├── static/
│   ├── index.html      # Frontend HTML
│   ├── styles.css      # Premium dark theme CSS
│   └── app.js          # Frontend JavaScript
├── uploads/            # Uploaded images (auto-created)
├── results/            # Detection results (auto-created)
└── README.md
```

---

### Limitations

- Detects unusual visual/spectral patterns; does **not** classify what the anomaly is  
- Detection accuracy depends on image quality, lighting conditions, and anomaly characteristics  
- Large images are automatically resized to 1024px max dimension for performance  
- False detections may occur due to shadows, lighting changes, or background variations  
