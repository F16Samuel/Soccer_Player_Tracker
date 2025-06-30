
# ⚽ Football Player Tracking with Radar Projection

This project implements an advanced player tracking system for football video analytics. Using DeepSORT the program tracks multiple objects in a given soccer video

---

## 📁 Project Structure

```
root/
│
├── ml_app/
│   ├── classification.py         # Detection, team classification, tracking, and annotation pipeline
│   ├── config.py                 # Constants for IDs and paths
│   ├── pipeline.py               # Entry point for running the classification
│   ├── tracking.py               # DeepSORT and custom tracking modules
│   ├── utils.py                  # Helper utilities and team classifier
│
├── models/
│   ├── best.pt                   # YOLOv8 player detection model
│   └── team_classifier_inf.pkl   # Pretrained team classifier (SigLIP/Transformer)
│
├── videos/
│   ├── input/                    # Input football match video
│   └── output/                   # Output video with annotations
│
├── .gitignore
├── LICENSE
├── main.py                       # Wrapper Script
└── requirements.txt              # Python dependencies
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/F16Samuel/soccer_player_tracker.git
cd football-tracking/ml_app
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # For Unix/Mac
venv\Scripts\activate         # For Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ensure Required Models are Available

Place the following files in the `models/` directory:
- `best.pt` — Trained YOLOv8 model for detecting players, goalkeepers, referees, and the ball.
- `team_classifier_inf.pkl` — Pretrained classifier for jersey/team ID classification.

---

## ▶️ Running the Code

### Run via CLI

If you've extended it into `main.py`, you can run:

```bash
python main.py --input videos/input/your_video.mp4 --output videos/output/your_video_out.mp4
```

---

## ⚙️ Environment & Dependencies

Tested on:
- Python 3.12.5
- CUDA 11.8 (if running with GPU)
- Torch ≥ 2.0
- ultralytics (for YOLOv11)
- supervision
- deep_sort_realtime
- opencv-python
- numpy
- pandas
- matplotlib
- transformers
- scipy
- umap-learn
- tqdm
- more-itertools

All dependencies are listed in `requirements.txt`.

**To install with GPU support (optional):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📽️ Output

The processed video includes:
- Elliptical bounding boxes with unique tracker IDs
- Ball tracking with triangle markers
- Annotated team classification (color-coded)
- Radar-style projection (future enhancement)

---

## 💡 Notes

- The project supports plug-and-play tracking modules.
- Radar-based field-aware tracker offers re-identification even after players re-enter the frame, but it’s still under tuning.
- Video must be 720p for consistent detection/classification.

---

## 📌 Future Improvements

- Multi-camera tracker ID consistency using radar mapping.
- Jersey number OCR using ResNet or YOLO OCR head.
- Full analytics dashboard: heatmaps, possession zones, player movement trails.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Deep SORT RealTime](https://github.com/levan92/deep_sort_realtime)
- [Supervision](https://github.com/roboflow/supervision)

---
