import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional, Tuple


class FaceDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.detection_confidence = config.video.detection_confidence
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        detection_map = self.detection_head(features)
        bbox_map = self.bbox_head(features)
        
        batch_size = x.size(0)
        faces = []
        
        for b in range(batch_size):
            det_map = detection_map[b, 0]
            
            y_indices, x_indices = torch.where(det_map > self.detection_confidence)
            
            if len(y_indices) == 0:
                faces.append(torch.zeros(1, 4).to(x.device))
            else:
                top_k = min(len(y_indices), self.config.video.max_faces)
                scores = det_map[y_indices, x_indices]
                top_indices = torch.topk(scores, top_k).indices
                
                y_sel = y_indices[top_indices]
                x_sel = x_indices[top_indices]
                
                bboxes = bbox_map[b, :, y_sel, x_sel].T
                bboxes = torch.sigmoid(bboxes) * x.size(-1)
                faces.append(bboxes)
        
        return faces


class LipEncoder(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=256):
        super().__init__()
        
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        
        self.temporal_conv = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        batch_size, _, time_steps = x.shape[:3]
        
        x = self.conv3d_layers(x)
        
        x = rearrange(x, 'b c t h w -> b c t (h w)')
        x = torch.mean(x, dim=-1)
        
        x = self.temporal_conv(x)
        
        x = rearrange(x, 'b c t -> b t c')
        x = self.projection(x)
        
        return x


class VideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.face_detector = FaceDetector(config)
        self.lip_encoder = LipEncoder(
            input_dim=config.video.lip_size[0],
            hidden_dim=config.model.video_encoder_dim
        )
        
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, 100, config.model.video_encoder_dim) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.video_encoder_dim,
            nhead=config.model.video_encoder_heads,
            dim_feedforward=config.model.video_encoder_ffn_dim,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.video_encoder_layers
        )
        
        self.layer_norm = nn.LayerNorm(config.model.video_encoder_dim)
        self.speaker_embedding = nn.Embedding(
            config.model.max_speakers,
            config.model.video_encoder_dim
        )
        
    def extract_lip_regions(self, frames, faces):
        batch_size, time_steps = frames.shape[:2]
        lip_regions = []
        
        for b in range(batch_size):
            face_bboxes = faces[b]
            batch_lips = []
            
            for t in range(time_steps):
                frame = frames[b, t]
                
                if face_bboxes.size(0) > 0:
                    bbox = face_bboxes[0]
                    
                    x1, y1, w, h = bbox.int()
                    x2, y2 = x1 + w, y1 + h
                    
                    y_center = (y1 + y2) // 2
                    lip_y1 = y_center
                    lip_y2 = min(y2, lip_y1 + h // 3)
                    
                    lip_region = frame[:, lip_y1:lip_y2, x1:x2]
                    
                    if lip_region.numel() > 0:
                        lip_region = F.interpolate(
                            lip_region.unsqueeze(0),
                            size=self.config.video.lip_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    else:
                        lip_region = torch.zeros(3, *self.config.video.lip_size).to(frame.device)
                else:
                    lip_region = torch.zeros(3, *self.config.video.lip_size).to(frame.device)
                
                lip_region = torch.mean(lip_region, dim=0, keepdim=True)
                batch_lips.append(lip_region)
            
            batch_lips = torch.stack(batch_lips)
            lip_regions.append(batch_lips)
        
        return torch.stack(lip_regions)
    
    def forward(self, video_frames, speaker_ids=None):
        batch_size, time_steps = video_frames.shape[:2]
        
        first_frame = video_frames[:, 0]
        faces = self.face_detector(first_frame)
        
        lip_regions = self.extract_lip_regions(video_frames, faces)
        
        visual_features = self.lip_encoder(lip_regions)
        
        if visual_features.size(1) < 100:
            visual_features = visual_features + self.temporal_embedding[:, :visual_features.size(1)]
        else:
            visual_features = visual_features + self.temporal_embedding
        
        if speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            visual_features = visual_features + speaker_emb.unsqueeze(1)
        
        visual_features = self.transformer(visual_features)
        visual_features = self.layer_norm(visual_features)
        
        return visual_features, faces