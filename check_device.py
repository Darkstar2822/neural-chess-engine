#!/usr/bin/env python3
import torch
from config import Config

def check_device_support():
    print("🔍 Checking device support...")
    print(f"PyTorch version: {torch.__version__}")
    
    print(f"\n📱 MPS (M1/M2) available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("✅ M1/M2 GPU acceleration supported!")
    
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
    
    print(f"\n🎯 Selected device: {Config.DEVICE}")
    
    # Test tensor operations
    try:
        if Config.DEVICE == "mps":
            x = torch.randn(1000, 1000).to(Config.DEVICE)
            y = torch.matmul(x, x)
            print("✅ MPS tensor operations working!")
        elif Config.DEVICE == "cuda":
            x = torch.randn(1000, 1000).to(Config.DEVICE)
            y = torch.matmul(x, x)
            print("✅ CUDA tensor operations working!")
        else:
            print("ℹ️  Using CPU")
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        print("Falling back to CPU")

if __name__ == "__main__":
    check_device_support()