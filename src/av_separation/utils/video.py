import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Union
from pathlib import Path
import warnings
import mediapipe as mp


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.target_fps = config.fps
        self.image_size = config.image_size
        self.face_size = config.face_size
        
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=config.detection_confidence
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.max_faces,
            refine_landmarks=True,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence
        )
    
    def load_video(
        self,
        file_path: Union[str, Path],
        max_frames: Optional[int] = None
    ) -> np.ndarray:
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        cap = cv2.VideoCapture(str(file_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {file_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        frame_interval = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and (max_frames is None or len(frames) < max_frames):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame = cv2.resize(frame, self.image_size)
                
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise RuntimeError("No frames extracted from video")
        
        return np.array(frames)
    
    def save_video(
        self,
        frames: np.ndarray,
        file_path: Union[str, Path],
        fps: Optional[int] = None
    ):
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps or self.target_fps
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def detect_faces(
        self,
        frame: np.ndarray
    ) -> List[dict]:
        
        results = self.face_detection.process(frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                faces.append({
                    'bbox': [x, y, width, height],
                    'confidence': detection.score[0] if detection.score else 0.0,
                    'landmarks': self._extract_landmarks(detection)
                })
        
        return faces
    
    def extract_lip_region(
        self,
        frame: np.ndarray,
        face_bbox: List[int],
        expand_ratio: float = 0.2
    ) -> np.ndarray:
        
        x, y, w, h = face_bbox
        
        y_center = y + h // 2
        lip_y = int(y_center + h * 0.1)
        lip_height = int(h * 0.3)
        
        lip_x = x + int(w * 0.2)
        lip_width = int(w * 0.6)
        
        expand_x = int(lip_width * expand_ratio)
        expand_y = int(lip_height * expand_ratio)
        
        lip_x = max(0, lip_x - expand_x)
        lip_y = max(0, lip_y - expand_y)
        lip_width = min(frame.shape[1] - lip_x, lip_width + 2 * expand_x)
        lip_height = min(frame.shape[0] - lip_y, lip_height + 2 * expand_y)
        
        lip_region = frame[lip_y:lip_y+lip_height, lip_x:lip_x+lip_width]
        
        if lip_region.size == 0:
            lip_region = np.zeros((*self.config.lip_size, 3), dtype=np.uint8)
        else:
            lip_region = cv2.resize(lip_region, self.config.lip_size)
        
        return lip_region
    
    def extract_face_landmarks(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        
        results = self.face_mesh.process(frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            h, w = frame.shape[:2]
            landmarks_array = []
            
            for landmark in landmarks.landmark:
                landmarks_array.append([
                    landmark.x * w,
                    landmark.y * h,
                    landmark.z * w
                ])
            
            return np.array(landmarks_array)
        
        return None
    
    def track_faces(
        self,
        frames: np.ndarray
    ) -> List[List[dict]]:
        
        all_detections = []
        
        for frame in frames:
            faces = self.detect_faces(frame)
            all_detections.append(faces)
        
        tracked_faces = self._apply_tracking(all_detections)
        
        return tracked_faces
    
    def _apply_tracking(
        self,
        detections: List[List[dict]]
    ) -> List[List[dict]]:
        
        if len(detections) == 0:
            return []
        
        tracked = []
        active_tracks = []
        next_id = 0
        
        for frame_idx, frame_detections in enumerate(detections):
            matched_tracks = []
            
            for detection in frame_detections:
                best_match = None
                best_iou = 0.0
                
                for track in active_tracks:
                    iou = self._compute_iou(
                        detection['bbox'],
                        track['last_bbox']
                    )
                    
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_match = track
                
                if best_match:
                    best_match['last_bbox'] = detection['bbox']
                    best_match['frames'].append(frame_idx)
                    detection['track_id'] = best_match['id']
                    matched_tracks.append(best_match)
                else:
                    new_track = {
                        'id': next_id,
                        'last_bbox': detection['bbox'],
                        'frames': [frame_idx]
                    }
                    detection['track_id'] = next_id
                    matched_tracks.append(new_track)
                    next_id += 1
            
            active_tracks = matched_tracks
            tracked.append(frame_detections)
        
        return tracked
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _extract_landmarks(self, detection) -> Optional[dict]:
        
        if not hasattr(detection, 'location_data'):
            return None
        
        landmarks = {}
        keypoints = detection.location_data.relative_keypoints
        
        if len(keypoints) >= 6:
            landmarks['right_eye'] = (keypoints[0].x, keypoints[0].y)
            landmarks['left_eye'] = (keypoints[1].x, keypoints[1].y)
            landmarks['nose_tip'] = (keypoints[2].x, keypoints[2].y)
            landmarks['mouth_center'] = (keypoints[3].x, keypoints[3].y)
            landmarks['right_ear'] = (keypoints[4].x, keypoints[4].y)
            landmarks['left_ear'] = (keypoints[5].x, keypoints[5].y)
        
        return landmarks if landmarks else None
    
    def augment_video(
        self,
        frames: np.ndarray,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip: bool = True
    ) -> np.ndarray:
        
        augmented = frames.copy()
        
        if np.random.random() < 0.5:
            brightness = np.random.uniform(*brightness_range)
            augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)
        
        if np.random.random() < 0.5:
            contrast = np.random.uniform(*contrast_range)
            mean = np.mean(augmented, axis=(1, 2, 3), keepdims=True)
            augmented = np.clip((augmented - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        if horizontal_flip and np.random.random() < 0.5:
            augmented = np.flip(augmented, axis=2)
        
        return augmented