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

from tracking import create_deepsort_tracker
from utils import resolve_goalkeepers_team_id
from utils import team_classifier as tc
from config import BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID, PLAYER_DETECTION_MODEL

# === Main Processing Pipeline ===
def process_soccer_video(source_path: str, output_path: str) -> None:
    """
    Main processing pipeline with enhanced tracking capabilities
    """
    # Load models (these need to be loaded before calling this function)
    print("🏈 Starting Soccer Video Analysis with Enhanced Tracking...")
    
    # Initialize tracking system
    tracker = create_deepsort_tracker()
    team_classifier = tc()
    
    # Setup annotators
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )
    
    # Setup video processing
    video_info = sv.VideoInfo.from_video_path(source_path)
    frame_generator = sv.get_video_frames_generator(source_path)
    
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_info.fps,
        (1280, 720)
    )
    
    print(f"📹 Processing {video_info.total_frames} frames...")
    
    try:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
            # === DETECTION PHASE ===
            result = PLAYER_DETECTION_MODEL(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Process ball detections
            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            
            # Process other detections
            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            
            # Separate detection types
            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]
            
            # === TEAM CLASSIFICATION ===
            if len(players_detections) == 0:
                continue

            if len(players_detections) > 0:
                players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                try:
                    players_detections.class_id = team_classifier.predict(players_crops)
                except:
                    # Fallback if team classifier fails
                    players_detections.class_id = np.zeros(len(players_detections), dtype=int)
            
            # Resolve goalkeeper teams
            if len(goalkeepers_detections) > 0 and len(players_detections) > 0:
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                    players_detections, goalkeepers_detections)
            
            # Prepare combined detections for tracking
            combined_detections = sv.Detections.merge([players_detections, goalkeepers_detections])
            combined_team_ids = np.concatenate([
                players_detections.class_id if len(players_detections) > 0 else np.array([]),
                goalkeepers_detections.class_id if len(goalkeepers_detections) > 0 else np.array([])
            ]).astype(int)
            
            # === ENHANCED TRACKING ===
            if len(combined_detections) > 0:
                tracker_ids = tracker.update_trackers(
                    combined_detections, combined_team_ids)
                if tracker_ids is None:
                    print(f"⚠️ Warning: tracker.update_trackers returned None at frame {frame_idx}")
                    tracker_ids = np.array([-1] * len(combined_detections))  # fallback dummy IDs

                # Update detection tracker IDs
                players_end_idx = len(players_detections)
                if players_end_idx > 0:
                    players_detections.tracker_id = tracker_ids[:players_end_idx]
                if len(goalkeepers_detections) > 0:
                    goalkeepers_detections.tracker_id = tracker_ids[players_end_idx:]
            
            # Process referee detections
            if len(referees_detections) > 0:
                referees_detections.class_id = np.full(len(referees_detections), 2, dtype=int)
                referees_detections.tracker_id = np.arange(
                    10000, 10000 + len(referees_detections))  # Separate ID range for referees
            
            # === FRAME ANNOTATIONS ===
            # Merge all detections for annotation
            all_detections = sv.Detections.merge([
                players_detections, goalkeepers_detections, referees_detections])
            
            # Create labels with tracker IDs
            labels = []
            if tracker_ids is None:
                print(f"⚠️ Warning: tracker.update_trackers returned None at frame {frame_idx}")
                tracker_ids = np.array([-1] * len(combined_detections))  # fallback dummy IDs

            for i, tracker_id in enumerate(all_detections.tracker_id):
                if tracker_id is not None:
                    labels.append(f"#{tracker_id}")
                else:
                    labels.append(f"#?")
            
            # Ensure class_id is int type
            all_detections.class_id = all_detections.class_id.astype(int)
            
            # === FRAME 1: Detection View ===
            detection_view = frame.copy()
            detection_view = ellipse_annotator.annotate(detection_view, all_detections)
            detection_view = label_annotator.annotate(detection_view, all_detections, labels)
            detection_view = triangle_annotator.annotate(detection_view, ball_detections)
            
            # Add floating points visualization on detection view
            for fp in tracker.floating_points:
                cv2.circle(detection_view, 
                          (int(fp.last_position[0]), int(fp.last_position[1])), 
                          8, (0, 255, 255), 2)  # Yellow circles for floating points
                cv2.putText(detection_view, f"#{fp.tracker_id}", 
                           (int(fp.last_position[0]) + 10, int(fp.last_position[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add tracking info overlay
            info_text = [
                f"Frame: {frame_idx + 1}/{video_info.total_frames}",
                f"Active: {len(tracker.active_trackers)}",
                f"Floating: {len(tracker.floating_points)}",
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(detection_view, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
                
            video_writer.write(detection_view)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # === Release Video Writer ===
        video_writer.release()
        print(f"✅ Video processing complete! Output saved to: {output_path}")
        
        # Print final statistics
        print("\n=== FINAL TRACKING STATISTICS ===")
        print(f"Total frames processed: {frame_idx + 1}")
        print(f"Final active trackers: {len(tracker.active_trackers)}")
        print(f"Final floating points: {len(tracker.floating_points)}")
        
        # Get highest tracker ID used (for DeepSORT, IDs are managed internally)
        if tracker.active_trackers:
            highest_id = max(tracker.active_trackers.keys())
            print(f"Highest tracker ID used: {highest_id}")
        else:
            print("No active trackers at end of processing")
        
        # Print tracker details
        print("\nActive Trackers:")
        if tracker.active_trackers:
            for tid, player_tracker in tracker.active_trackers.items():
                role = "GK" if player_tracker.is_goalkeeper else "Player"
                team = player_tracker.team_id
                pos = player_tracker.position
                conf = player_tracker.confidence
                print(f"  #{tid}: {role} (Team {team}) at ({pos[0]:.1f}, {pos[1]:.1f}) [conf: {conf:.2f}]")
        else:
            print("  No active trackers")
        
        # Print floating points
        if tracker.floating_points:
            print("\nFloating Points:")
            for fp in tracker.floating_points:
                pos = fp.last_position
                print(f"  #{fp.tracker_id}: Team {fp.team_id} at ({pos[0]:.1f}, {pos[1]:.1f}), absent for {fp.frames_absent} frames")
        else:
            print("\nNo floating points")
        
        # Print goalkeeper summary
        if tracker.goalkeeper_trackers:
            print("\nGoalkeeper Assignments:")
            for team_id, gk_tracker_id in tracker.goalkeeper_trackers.items():
                if gk_tracker_id in tracker.active_trackers:
                    gk = tracker.active_trackers[gk_tracker_id]
                    pos = gk.position
                    print(f"  Team {team_id}: Tracker #{gk_tracker_id} at ({pos[0]:.1f}, {pos[1]:.1f})")
                else:
                    print(f"  Team {team_id}: Goalkeeper tracker #{gk_tracker_id} (inactive)")
        
        # Print DeepSORT specific statistics
        print("\n=== DEEPSORT STATISTICS ===")
        print(f"Embedder model: {tracker.deepsort.embedder}")
        
        # Print tracking performance summary
        total_detections = frame_idx + 1  # Approximate
        if total_detections > 0:
            avg_trackers_per_frame = len(tracker.active_trackers)  # Final count as approximation
            print(f"Average trackers per frame: ~{avg_trackers_per_frame}")
        
        # Memory cleanup
        print("\n=== CLEANUP ===")
        print("Releasing DeepSORT resources...")
        # Note: DeepSORT handles its own cleanup, but we can reset if needed
        # tracker.reset()  # Uncomment if you want to reset tracker state
        
        print("Processing completed successfully! 🎉")