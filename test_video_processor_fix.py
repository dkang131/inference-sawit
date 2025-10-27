#!/usr/bin/env python3
"""
Test script to verify video processor fix
"""

import os
import tempfile
import cv2
import numpy as np
import time
from utils.video_processor import VideoProcess


def create_test_video(duration_seconds=10, fps=30, resolution=(640, 480)):
    """Create a simple test video"""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, f'test_video_{duration_seconds}s.mp4')
    
    total_frames = duration_seconds * fps
    width, height = resolution
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
    
    for frame_num in range(total_frames):
        # Create frame with moving content
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(height):
            frame[y, :, 0] = int((y / height) * 255)  # Red gradient
            frame[y, :, 1] = int(((height - y) / height) * 255)  # Green gradient
        
        # Add moving objects
        circle_x = int((frame_num / total_frames) * width)
        circle_y = height // 2
        cv2.circle(frame, (circle_x, circle_y), 20, (255, 255, 255), -1)
        
        # Add frame number text
        cv2.putText(frame, f'Frame {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    return video_path, temp_dir


def test_video_processor():
    """Test the video processor with a simple video"""
    print("Creating test video...")
    video_path, temp_dir = create_test_video(duration_seconds=5, fps=30, resolution=(640, 480))
    
    try:
        print(f"Test video created: {video_path}")
        
        # Initialize video processor
        processor = VideoProcess()
        
        # Start processing
        print("Starting video processing...")
        success, message = processor.start_processing(video_path, '11l-150-hyper.pt')
        
        if success:
            print(f"Processing started successfully: {message}")
            
            # Monitor processing for a few seconds
            for i in range(10):
                time.sleep(1)
                result = processor.get_latest_result()
                if result:
                    print(f"Progress: {result.get('frame_count', 0)} frames processed")
                else:
                    print("No result available yet")
                
                if processor.is_processing_complete():
                    print("Processing completed!")
                    break
            
            # Stop processing
            print("Stopping processing...")
            stop_result = processor.stop()
            print(f"Stop result: {stop_result}")
            
        else:
            print(f"Failed to start processing: {message}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == '__main__':
    print("Testing video processor fix...")
    success = test_video_processor()
    if success:
        print("✓ Video processor test passed!")
    else:
        print("✗ Video processor test failed!")