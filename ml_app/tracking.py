# === DeepSORT Tracking Classes ===

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
from utils import team_classifier

@dataclass
class FloatingPoint:
    """Represents a player who has left the frame"""
    last_position: Tuple[float, float]
    tracker_id: int
    team_id: int
    frames_absent: int = 0
    
@dataclass 
class PlayerTracker:
    """Enhanced player tracker with DeepSORT integration"""
    tracker_id: int
    team_id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    is_goalkeeper: bool = False
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x1, y1, x2, y2
    
class DeepSORTTracker:
    """Handles player tracking using DeepSORT algorithm"""
    
    def __init__(self, max_age: int = 70, n_init: int = 3, max_iou_distance: float = 0.7,
                 max_cosine_distance: float = 0.2, nn_budget: int = 2500):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum number of frames to keep alive a track without associated detections
            n_init: Number of consecutive detections before the track is confirmed
            max_iou_distance: Maximum IoU distance for matching
            max_cosine_distance: Maximum cosine distance for feature matching
            nn_budget: Maximum size of the appearance descriptor gallery
        """
        self.deepsort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None,
            embedder="mobilenet",  # Can be 'mobilenet', 'clip_RN50', 'clip_RN101', 'clip_ViT-B/32'
            half=True,
            bgr=True,
            embedder_gpu=False,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        self.active_trackers: Dict[int, PlayerTracker] = {}
        self.floating_points: List[FloatingPoint] = []
        self.frame_count = 0
        self.goalkeeper_positions = [(100, 300), (500, 300)]  # Default goalkeeper positions
        self.goalkeeper_trackers: Dict[int, int] = {}  # team_id -> tracker_id mapping
        
        # Track previous positions for velocity calculation
        self.previous_positions: Dict[int, Tuple[float, float]] = {}
        
    def initialize_goalkeepers(self, detections: sv.Detections, team_ids: np.ndarray):
        """Initialize goalkeeper trackers at known positions"""
        goalkeeper_mask = detections.class_id == 0  # Assuming 0 is goalkeeper class
        
        if not np.any(goalkeeper_mask):
            # If no goalkeepers detected, create virtual detections at known positions
            for i, pos in enumerate(self.goalkeeper_positions):
                # Create a virtual bounding box around goalkeeper position
                x_center, y_center = pos
                bbox_width, bbox_height = 80, 120  # Typical goalkeeper size
                
                virtual_bbox = [
                    x_center - bbox_width//2,  # x1
                    y_center - bbox_height//2,  # y1
                    x_center + bbox_width//2,   # x2
                    y_center + bbox_height//2   # y2
                ]
                
                tracker = PlayerTracker(
                    tracker_id=f"GK_{i}",
                    team_id=i,
                    position=pos,
                    is_goalkeeper=True,
                    confidence=0.8,
                    bbox=tuple(virtual_bbox)
                )
                self.active_trackers[f"GK_{i}"] = tracker
                self.goalkeeper_trackers[i] = f"GK_{i}"
    
    def prepare_detections_for_deepsort(self, detections: sv.Detections, 
                                      team_ids: np.ndarray, frame: np.ndarray) -> List:
        """
        Convert supervision detections to DeepSORT format
        
        Returns:
            List of [bbox, confidence, class_id, team_id] for each detection
        """
        deepsort_detections = []
        
        if len(detections) == 0:
            return deepsort_detections
        
        # Validate frame input
        if not isinstance(frame, np.ndarray):
            print(f"⚠️  Warning: Frame is not a numpy array, type: {type(frame)}")
            # Use default frame dimensions if frame is invalid
            h, w = 720, 1280  # Default video dimensions
        else:
            h, w = frame.shape[:2]
        
        # Get bounding boxes in xyxy format
        bboxes = detections.xyxy
        confidences = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        class_ids = detections.class_id if detections.class_id is not None else np.zeros(len(detections))
        
        for i, (bbox, conf, class_id, team_id) in enumerate(zip(bboxes, confidences, class_ids, team_ids)):
            # Convert bbox to [x1, y1, x2, y2] format
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within frame bounds
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(x1+1, min(w, x2))
            y2 = max(y1+1, min(h, y2))
            
            deepsort_detections.append([
                [x1, y1, x2, y2],  # bbox
                float(conf),        # confidence
                int(class_id),      # class_id
                int(team_id)        # team_id (custom field)
            ])
        
        return deepsort_detections
    
    def extract_player_features(self, frame: np.ndarray, bboxes: List) -> np.ndarray:
        """
        Extract appearance features for each detection
        This is handled internally by DeepSORT's embedder
        """
        # DeepSORT handles feature extraction internally
        # We just need to provide the frame and bounding boxes
        return None
    
    def update_trackers(self, detections: sv.Detections, team_ids: np.ndarray, 
                       frame: np.ndarray = None) -> np.ndarray:
        """
        Update trackers with new detections using DeepSORT
        
        Args:
            detections: Supervision detections
            team_ids: Team ID for each detection
            frame: Current frame for feature extraction (optional)
            
        Returns:
            Tracker IDs for each detection
        """
        self.frame_count += 1
        
        # Initialize goalkeepers on first frame
        if self.frame_count == 1:
            self.initialize_goalkeepers(detections, team_ids)
        
        # Handle missing or invalid frame
        if frame is None or not isinstance(frame, np.ndarray):
            # print(f"⚠️  Warning: Invalid frame provided (type: {type(frame)}). Using fallback tracking.")
            # Fallback to simple tracking without deep features
            return self._fallback_tracking(detections, team_ids)
        
        # Prepare detections for DeepSORT
        deepsort_detections = self.prepare_detections_for_deepsort(detections, team_ids, frame)
        
        if len(deepsort_detections) == 0:
            # No detections to process
            return np.zeros(len(detections), dtype=int)
        
        try:
            # Update DeepSORT tracker
            tracks = self.deepsort.update_tracks(deepsort_detections, frame=frame)
        except Exception as e:
            return self._fallback_tracking(detections, team_ids)
        
        # Process tracking results
        result_tracker_ids = np.zeros(len(detections), dtype=object)
        detection_idx = 0
        
        # Update active trackers and assign IDs
        current_active_trackers = {}
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            
            # Calculate position (bottom center of bbox)
            x_center = (bbox[0] + bbox[2]) / 2
            y_bottom = bbox[3]
            position = (float(x_center), float(y_bottom))
            
            # Calculate velocity if we have previous position
            velocity = (0.0, 0.0)
            if track_id in self.previous_positions:
                prev_pos = self.previous_positions[track_id]
                velocity = (position[0] - prev_pos[0], position[1] - prev_pos[1])
            
            # Determine team_id from detection data
            team_id = 0  # Default
            if detection_idx < len(deepsort_detections):
                team_id = deepsort_detections[detection_idx][3] if len(deepsort_detections[detection_idx]) > 3 else 0
            
            # Check if this is a goalkeeper
            is_goalkeeper = track_id in self.goalkeeper_trackers.values()
            
            # Create/update tracker
            tracker = PlayerTracker(
                tracker_id=track_id,
                team_id=team_id,
                position=position,
                velocity=velocity,
                confidence=float(track.get_det_conf() if hasattr(track, 'get_det_conf') else 1.0),
                is_goalkeeper=is_goalkeeper,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            )
            
            current_active_trackers[track_id] = tracker
            self.previous_positions[track_id] = position
            
            # Find corresponding detection index
            if detection_idx < len(result_tracker_ids):
                result_tracker_ids[detection_idx] = track_id
                detection_idx += 1
        
        # Update active trackers
        self.active_trackers = current_active_trackers
        
        # Handle floating points (tracks that are no longer active)
        active_track_ids = set(self.active_trackers.keys())
        previous_track_ids = set(self.previous_positions.keys())
        
        for track_id in previous_track_ids - active_track_ids:
            if track_id in self.previous_positions:
                # Move to floating points if not a goalkeeper
                if track_id not in self.goalkeeper_trackers.values():
                    last_pos = self.previous_positions[track_id]
                    # Try to determine team_id from previous tracker data
                    team_id = 0  # Default
                    
                    fp = FloatingPoint(
                        last_position=last_pos,
                        tracker_id=track_id,
                        team_id=team_id,
                        frames_absent=0
                    )
                    self.floating_points.append(fp)
                
                # Clean up previous positions for non-goalkeeper tracks
                if track_id not in self.goalkeeper_trackers.values():
                    del self.previous_positions[track_id]
        
        # Update floating points and remove expired ones
        self.floating_points = [
            FloatingPoint(fp.last_position, fp.tracker_id, fp.team_id, fp.frames_absent + 1)
            for fp in self.floating_points
            if fp.frames_absent < 300  # Timeout after 300 frames
        ]
        
        # Convert object array to int array, handling any remaining None values
        final_result = np.zeros(len(detections), dtype=int)
        for i, track_id in enumerate(result_tracker_ids):
            if track_id is not None:
                final_result[i] = int(track_id)
            else:
                final_result[i] = 0  # Default for untracked detections
        
    def _fallback_tracking(self, detections: sv.Detections, team_ids: np.ndarray) -> np.ndarray:
        """
        Fallback tracking method when DeepSORT fails or frame is invalid
        """
        if len(detections) == 0:
            return np.zeros(0, dtype=int)
        
        detection_positions = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        result_tracker_ids = np.zeros(len(detections), dtype=int)
        
        # Simple distance-based matching for fallback
        max_distance = 100.0  # pixels
        
        for i, (pos, team_id) in enumerate(zip(detection_positions, team_ids)):
            best_match = None
            min_distance = float('inf')
            
            # Check existing active trackers
            for tid, tracker in self.active_trackers.items():
                if tracker.team_id == team_id:  # Same team
                    distance = np.sqrt((pos[0] - tracker.position[0])**2 + 
                                     (pos[1] - tracker.position[1])**2)
                    if distance < max_distance and distance < min_distance:
                        min_distance = distance
                        best_match = tid
            
            if best_match is not None:
                # Update existing tracker
                result_tracker_ids[i] = best_match
                self.active_trackers[best_match].position = (float(pos[0]), float(pos[1]))
            else:
                # Create new tracker
                new_id = max(self.active_trackers.keys(), default=0) + 1
                tracker = PlayerTracker(
                    tracker_id=new_id,
                    team_id=int(team_id),
                    position=(float(pos[0]), float(pos[1]))
                )
                self.active_trackers[new_id] = tracker
                result_tracker_ids[i] = new_id
        
        return result_tracker_ids
    
    def get_track_history(self, track_id: int, max_length: int = 300) -> List[Tuple[float, float]]:
        """
        Get position history for a specific track
        
        Args:
            track_id: ID of the track
            max_length: Maximum number of historical positions to return
            
        Returns:
            List of (x, y) positions
        """
        # This would need to be implemented with additional history storage
        # For now, return current position if tracker exists
        if track_id in self.active_trackers:
            return [self.active_trackers[track_id].position]
        return []
    
    def reset(self):
        """Reset the tracker state"""
        self.deepsort = DeepSort(
            max_age=70,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.2,
            nn_budget=100,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        self.active_trackers.clear()
        self.floating_points.clear()
        self.frame_count = 0
        self.previous_positions.clear()
        self.goalkeeper_trackers.clear()

# === Usage ===

def create_deepsort_tracker():
    """
    Factory function to create a DeepSORT tracker with optimized settings
    """
    return DeepSORTTracker(
        max_age=50,        # Reduce for faster cleanup of lost tracks
        n_init=2,          # Reduce for faster track initialization
        max_iou_distance=0.7,
        max_cosine_distance=0.3,  # Slightly higher for sports scenarios
        nn_budget=150      # Larger budget for better re-identification
    )