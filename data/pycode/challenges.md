# Code Generation Challenges

Total: 10 challenges

---

## Challenge 1: CNN (easy)

**Prompt:** Create a simple 2-layer CNN that takes 64 input channels and outputs 128 channels

```python
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
```

---

## Challenge 2: CNN (medium)

**Prompt:** Create a residual block with skip connection, batch normalization, and ReLU activation

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
```

---

## Challenge 3: ATTENTION (medium)

**Prompt:** Create a multi-head self-attention layer with 8 heads and configurable embedding dimension

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, _ = x.shape
        
        q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)
```

---

## Challenge 4: CNN (hard)

**Prompt:** Create a bottleneck residual block used in ResNet-50 with 1x1, 3x3, 1x1 convolutions

```python
class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)
```

---

## Challenge 5: CNN (medium)

**Prompt:** Create a depthwise separable convolution block like in MobileNet

```python
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x
```

---

## Challenge 6: ATTENTION (medium)

**Prompt:** Create a squeeze-and-excitation block for channel attention

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y
```

---

## Challenge 7: TRANSFORMER (hard)

**Prompt:** Create a transformer encoder block with multi-head attention, layer norm, and feedforward network

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
```

---

## Challenge 8: ATTENTION (hard)

**Prompt:** Create a convolutional block attention module (CBAM) with both channel and spatial attention

```python
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        channel_att = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        x = x * channel_att
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att
```

---

## Challenge 9: CNN (hard)

**Prompt:** Create an inverted residual block like in MobileNetV2 with expansion, depthwise conv, and projection

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden, 1),
                nn.BatchNorm2d(hidden),
                nn.ReLU6()
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

---

## Challenge 10: RNN (medium)

**Prompt:** Create a simple GRU-based sequence model for text classification

```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)
```

---

