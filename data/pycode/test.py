  # """Build a test script that checks each level."""
        # return f'''

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

# Determinism and thread limits (injected from template)
SEED = {seed}
TORCH_THREADS = {torch_threads}
DEVICE = "{device}"
TIMEOUT_S = {timeout}

try:
    torch.set_num_threads(TORCH_THREADS)
except Exception:
    pass

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# User's generated code
{code}

# Test harness
{self.current_test}

def run_hierarchical_tests():
    try:
        # Find the main class defined
        model = None
        for name, obj in list(globals().items()):
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                # Try to instantiate with reasonable defaults
                try:
                    if 'channels' in name.lower() or 'conv' in name.lower():
                        model = obj(64, 128, 3)
                    elif 'attention' in name.lower():
                        model = obj(256)
                    elif 'transformer' in name.lower():
                        model = obj(256, 8)
                    else:
                        model = obj(64, 64)
                    break
                except:
                    continue
        
        if model is None:
            print("NO_MODEL_FOUND")
            return
            
        print("INSTANTIATE_OK")
        
        # Forward pass
        model.eval()
        if hasattr(model, 'conv1') or 'conv' in model.__class__.__name__.lower():
            x = torch.randn(2, 64, 32, 32)
        else:
            x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            out = model(x)
        print("FORWARD_OK")
        
        # Backward pass
        model.train()
        if hasattr(model, 'conv1') or 'conv' in model.__class__.__name__.lower():
            x = torch.randn(2, 64, 32, 32)
        else:
            x = torch.randn(2, 16, 256)
        out = model(x)
        loss = out.sum()
        loss.backward()
        print("BACKWARD_OK")
        
        # Stability: 10 training steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(10):
            if hasattr(model, 'conv1') or 'conv' in model.__class__.__name__.lower():
                x = torch.randn(2, 64, 32, 32)
            else:
                x = torch.randn(2, 16, 256)
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if torch.isnan(loss):
                print("NAN_DETECTED")
                return
        
        print("TRAIN_STABLE")
        
    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)

run_hierarchical_tests()