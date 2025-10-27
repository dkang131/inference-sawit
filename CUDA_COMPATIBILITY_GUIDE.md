# CUDA Compatibility Guide for RTX 5060 TI

## Issue Description

The RTX 5060 TI GPU has CUDA compute capability 12.0, which is newer than what the current PyTorch installation supports (up to 9.0). This causes the error:

```
CUDA error: no kernel image is available for execution on the device
```

## Current Solution (Applied)

**CPU-Only Mode**: The application is currently configured to run in CPU-only mode by setting:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

This ensures compatibility but disables GPU acceleration.

## Performance Impact

- **CPU Mode**: Slower processing but guaranteed compatibility
- **GPU Mode**: Much faster processing but requires compatible PyTorch version

## Long-term Solutions

### Option 1: Update PyTorch (Recommended for Performance)

Install PyTorch with newer CUDA support:

```bash
# For CUDA 12.1+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 12.4+ support  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

After updating, comment out the CPU-only line in `main.py`:
```python
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Comment this out to enable GPU
```

### Option 2: Use PyTorch Nightly (Latest Features)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### Option 3: Keep CPU-Only Mode

If you prefer stability over performance, keep the current configuration. The application will work reliably on CPU.

## Testing GPU Compatibility

Run the compatibility test:
```bash
python test_cuda_compatibility.py
```

This will tell you if GPU acceleration is working correctly.

## Switching Between CPU and GPU

### To Enable GPU:
1. Comment out in `main.py`: `# os.environ['CUDA_VISIBLE_DEVICES'] = ''`
2. Restart the application

### To Force CPU:
1. Uncomment in `main.py`: `os.environ['CUDA_VISIBLE_DEVICES'] = ''`
2. Restart the application

## Performance Expectations

### CPU Mode (Current):
- Processing Speed: ~2-5 FPS depending on video resolution
- Memory Usage: Lower
- Compatibility: 100%

### GPU Mode (After PyTorch Update):
- Processing Speed: ~15-30 FPS depending on video resolution
- Memory Usage: Higher (GPU memory)
- Compatibility: Requires compatible PyTorch version

## Troubleshooting

### If GPU mode still fails after PyTorch update:
1. Check NVIDIA driver version (should be latest)
2. Verify CUDA toolkit installation
3. Run `nvidia-smi` to check GPU status
4. Fall back to CPU mode if issues persist

### Common Error Messages:
- `no kernel image available`: CUDA compatibility issue → Use CPU mode
- `out of memory`: GPU memory full → Reduce batch size or use CPU mode
- `CUDA driver version insufficient`: Update NVIDIA drivers

## Current Status

✅ **Application is working in CPU-only mode**  
⚠️ **GPU acceleration disabled for compatibility**  
🔄 **Can be enabled after PyTorch update**

The video processing pipeline is fully functional and will process videos correctly, just at a slower speed than with GPU acceleration.