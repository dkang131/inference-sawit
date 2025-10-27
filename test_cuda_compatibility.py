#!/usr/bin/env python3
"""
Test CUDA compatibility for RTX 5060 TI and provide recommendations
"""

import torch
import sys
import os


def test_cuda_compatibility():
    """Test CUDA compatibility and provide recommendations"""
    print("=== CUDA Compatibility Test ===")
    
    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"Device {i}: {device_name}")
            print(f"  Compute Capability: {device_capability[0]}.{device_capability[1]}")
        
        # Test tensor operations
        try:
            print("\nTesting basic CUDA operations...")
            test_tensor = torch.randn(1, 3, 640, 640).cuda()
            result = test_tensor * 2
            print("✓ Basic CUDA tensor operations successful")
            
            # Test convolution (similar to what YOLO uses)
            print("Testing convolution operations...")
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            output = conv(test_tensor)
            print("✓ CUDA convolution operations successful")
            
            return True, "CUDA is working correctly"
            
        except Exception as e:
            print(f"✗ CUDA operations failed: {e}")
            return False, str(e)
    else:
        return False, "CUDA not available"


def get_recommendations(cuda_working, error_msg):
    """Provide recommendations based on test results"""
    print("\n=== Recommendations ===")
    
    if cuda_working:
        print("✓ CUDA is working correctly. You can use GPU acceleration.")
        print("  No changes needed to the application.")
    else:
        print("✗ CUDA issues detected.")
        print(f"  Error: {error_msg}")
        print("\nRecommended solutions:")
        print("1. Force CPU-only mode (safest option):")
        print("   - Uncomment the line in main.py: os.environ['CUDA_VISIBLE_DEVICES'] = ''")
        print("   - This will disable GPU acceleration but ensure compatibility")
        
        print("\n2. Update PyTorch for RTX 5060 TI compatibility:")
        print("   - Install PyTorch with CUDA 12.1 or newer:")
        print("   - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n3. Check NVIDIA driver version:")
        print("   - Ensure you have the latest NVIDIA drivers for RTX 5060 TI")
        print("   - Download from: https://www.nvidia.com/drivers/")
        
        print("\n4. Alternative: Use CPU-only PyTorch:")
        print("   - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")


def main():
    """Main test function"""
    try:
        cuda_working, error_msg = test_cuda_compatibility()
        get_recommendations(cuda_working, error_msg)
        
        if not cuda_working:
            print(f"\n=== Quick Fix ===")
            print("To immediately fix the issue and run the application:")
            print("1. Edit main.py")
            print("2. Uncomment this line: # os.environ['CUDA_VISIBLE_DEVICES'] = ''")
            print("3. Restart the application")
            print("This will run the application in CPU-only mode.")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()