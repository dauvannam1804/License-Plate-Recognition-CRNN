
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, dropout=0.2):
        super(CRNN, self).__init__()
        
        # CNN Backbone - Using ResNet18 for feature extraction
        # We need to modify it to preserve width for sequence modeling while reducing height to 1
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2]) # Remove fc and avgpool
        
        # Custom layers to adjust dimensions if needed.
        # ResNet18 downsamples by factor of 32 (5 maxpools/strided convs).
        # Input: (32, 128) -> Output: (512, 1, 4) if purely standard ResNet18?
        # Let's check:
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        # So output feature map is (512, 1, 4). 4 timesteps is too small for many labels.
        # We need to modify the strides in the later layers to keep width larger.
        
        # Re-implementing a simpler ResNet-like or VGG-like feature extractor might be safer/easier 
        # to ensure correct output dimensions for CRNN, or modifying ResNet strides.
        # Let's try modifying ResNet strides first.
        
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify strides of layer3 and layer4 to 1 for width (keep stride 2 for height?? No height is already small)
        # Actually input height 32 is small. 
        # Standard: 32 -> 16 (conv1) -> 8 (layer1 is stride 1) -> 4 (layer2) -> 2 (layer3) -> 1 (layer4)
        # Width 128 -> 64 -> 32 -> 16 -> 8 -> 4.
        
        # We want width to be e.g. 32 or 16 at least.
        # Let's change stride of layer3 and layer4.
        self.feature_extractor.layer3[0].conv1.stride = (2, 1) # Downsample H but not W?
        self.feature_extractor.layer3[0].downsample[0].stride = (2, 1)
        
        self.feature_extractor.layer4[0].conv1.stride = (2, 1) # Downsample H but not W
        self.feature_extractor.layer4[0].downsample[0].stride = (2, 1)
        
        # If we do (2,1) for layer3 and layer4:
        # H: 32conv1->16maxpool->16layer1->8layer2->4layer3->2layer4. Still 2.
        # We need H=1.
        # Maybe let's just use a simple Custom CNN. It is often less headache for 32x128 input.
        
        self.use_custom_cnn = True
        
        if self.use_custom_cnn:
            # VGG-Style backbone suitable for 32x128 input
            self.cnn_backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # 32x128 -> 16x64
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # 16x64 -> 8x32
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 8x32 -> 4x33 (padded?) OR use custom padding
                # Let's rely on standard CRNN paper arch or similar.
                # MaxPool (2,2) -> (2,1) -> (2,1) implies width stays larger.
                
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 4x32 -> 2x33
                
                nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU() # 2x32 -> 1x32 (approx) - Kernel 2 valid convolution reduces H from 2 to 1 if H=2.
            )
            # Output of above needs checking. 
            # Img: 32x128.
            # 1. Conv64, MP(2,2) -> 16x64
            # 2. Conv128, MP(2,2) -> 8x32
            # 3. Conv256, Conv256, MP((2,1) stride) -> H becomes 4, W becomes 32 (stride 1). 
            # 4. Conv512, Conv512, MP((2,1) stride) -> H becomes 2, W becomes 32.
            # 5. Conv512 (k=2, s=1, p=0) -> H becomes 1, W becomes 31.
            
            self.cnn_output_width = 512 # Channels
        
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True, dropout=dropout)
        self.embedding = nn.Linear(hidden_size * 2, vocab_size + 1) # +1 for blank

    def forward(self, x):
        # x: (B, 3, 32, 128)
        if hasattr(self, 'use_custom_cnn') and self.use_custom_cnn:
            features = self.cnn_backbone(x) # (B, 512, 1, W)
        else:
            # Not used
            pass
            
        # Reshape for RNN
        # (B, C, H, W) -> (B, C, 1, W) -> (B, C, W) -> (B, W, C)
        b, c, h, w = features.size()
        assert h == 1, f"Height must be 1, got {h}"
        features = features.squeeze(2) # (B, C, W)
        features = features.permute(0, 2, 1) # (B, W, C)
        
        # RNN
        rnn_out, _ = self.rnn(features) # (B, W, 2*hidden)
        
        # FC
        output = self.embedding(rnn_out) # (B, W, vocab+1)
        
        # Permute for CTC Loss (T, B, C)
        output = output.permute(1, 0, 2)
        
        return output
