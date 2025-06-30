# === Enhanced Soccer Player Tracking System ===
# Refactored with improved player ID tracking, re-identification, and goalkeeper initialization

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm
from collections import deque, defaultdict
from datetime import datetime
from more_itertools import chunked
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import heapq

from ultralytics import YOLO
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
import umap

from sports.common.team import TeamClassifier
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import (draw_points_on_pitch, draw_pitch_voronoi_diagram)

# === Constants ===
SOURCE_VIDEO_PATH = "videos/input/15sec_input_720p.mp4"
OUTPUT_VIDEO_PATH = "videos/output/15sec_input_720p.mp4"
DEVICE = 'cpu'
BATCH_SIZE = 32
BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3
REID_THRESHOLD = 0.5
STRIDE = 30
BUFFER_SIZE = 10000
CONFIG = SoccerPitchConfiguration()

# Tracking parameters
MAX_DISTANCE_THRESHOLD = 100  # Maximum distance for tracking association
FLOATING_POINT_TIMEOUT = 300   # Frames before floating point expires
GOALKEEPER_POSITIONS = [(100, 200), (500, 200)]  # Initial goalkeeper positions

PLAYER_DETECTION_MODEL = YOLO("models/best.pt")