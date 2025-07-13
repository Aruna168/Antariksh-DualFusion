import torch
import torch.nn as nn
import torch.nn.functional as F

class DualSRNet(nn.Module):
    """Dual Image Super-Resolution Network"""
    
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super(DualSRNet, self).__init__()
        
        # Feature extraction for both images
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, lr1, lr2):
        # Extract features from both images
        feat1 = self.feature_extractor(lr1)
        feat2 = self.feature_extractor(lr2)
        
        # Fuse features
        fused = torch.cat([feat1, feat2], dim=1)
        fused = self.fusion(fused)
        
        # Upsample to HR
        hr = self.upsampler(fused)
        
        return hr