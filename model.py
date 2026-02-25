import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # IMPORTANT: No BatchNorm for Super-Resolution (keeps details sharp)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out *= 0.1 # Residual scaling for stability
        out += residual
        return out

class SRModel(nn.Module):
    def __init__(self, upscale_factor=2, num_res_blocks=16):
        super(SRModel, self).__init__()
        self.upscale_factor = upscale_factor
        
        # 1. Feature Extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # 2. Deep Residual "Brain"
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 3. Final convolution before upscaling
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 4. Upsampling (PixelShuffle)
        self.conv_up = nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        # 5. Output mapping
        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Global Skip Connection (Bicubic Base)
        base = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # Residual Path
        x = self.conv1(x)
        res = x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x += res # Long skip connection
        
        x = self.pixel_shuffle(self.conv_up(x))
        detail = self.conv_final(x)
        
        return base + detail

if __name__ == "__main__":
    model = SRModel(upscale_factor=2)
    test_input = torch.randn(1, 3, 64, 64)
    test_output = model(test_input)
    print(f"Output shape: {test_output.shape}")
