# === Utility Functions ===
import numpy as np
import supervision as sv
import os
import pickle

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """Resolve goalkeeper team IDs based on proximity to team centroids"""
    if len(goalkeepers) == 0:
        return np.array([])
    
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    if len(players_xy) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)
    
    team_0_mask = players.class_id == 0
    team_1_mask = players.class_id == 1
    
    if not np.any(team_0_mask) or not np.any(team_1_mask):
        return np.zeros(len(goalkeepers_xy), dtype=int)
    
    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(goalkeepers_team_id)

def team_classifier():
    # Load the trained team classifier
    model_load_path = "models/team_classifier_inf.pkl"

    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Team classifier model not found at {model_load_path}. Please run training first.")

    with open(model_load_path, 'rb') as f:
        team_classifier = pickle.load(f)
    return team_classifier