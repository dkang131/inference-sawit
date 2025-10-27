#!/usr/bin/env python3
"""
Simple performance test to verify the performance monitoring functionality
"""

import time
import psutil
import tempfile
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from utils.video_processor import VideoProcess


class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.fps_samples = []
        self.processing_errors = []
        self.recovery_attempts = []
        
    def start_monitoring(self):
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.fps_samples = []
        self.processing_errors = []
        self.recovery_attempts = []
        
    def sample_performance(self, processor=None):
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            self.memory_samples.append({
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent
            })
            
            if processor and hasattr(processor, 'latest_result') and processor.latest_result:
                fps = processor.latest_result.get('fps', 0)
                self.fps_samples.append({
                    'timestamp': time.time(),
                    'fps': fps,
                    'frames_processed': processor.frames_processed
                })
                
        except Exception as e:
            print(f"Error sampling performance: {e}")
            
    def record_error(self, error_type, error_message, recovery_attempted=False):
        self.processing_errors.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_message,
            'recovery_attempted': recovery_attempted
        })
        
    def record_recovery_attempt(self, recovery_type, success=False):
        self.recovery_attempts.append({
            'timestamp': time.time(),
            'recovery_type': recovery_type,
            'success': success
        })
        
    def stop_monitoring(self):
        self.end_time = time.time()
        
    def get_performance_summary(self):
        if not self.memory_samples:
            return {
                'duration': 0,
                'peak_memory_mb': 0,
                'avg_memory_mb': 0,
                'avg_cpu_percent': 0,
                'avg_fps': 0,
                'peak_fps': 0,
                'total_errors': 0,
                'recovery_attempts': 0,
                'recovery_success_rate': 0
            }
        
        duration = (self.end_time or time.time()) - (self.start_time or 0)
        memory_values = [s['memory_mb'] for s in self.memory_samples]
        cpu_values = [s['cpu_percent'] for s in self.memory_samples if s['cpu_percent'] > 0]
        fps_values = [s['fps'] for s in self.fps_samples if s['fps'] > 0]
        
        successful_recoveries = sum(1 for r in self.recovery_attempts if r['success'])
        recovery_success_rate = (successful_recoveries / len(self.recovery_attempts) * 100) if self.recovery_attempts else 0
        
        return {
            'duration': round(duration, 2),
            'peak_memory_mb': round(max(memory_values) if memory_values else 0, 2),
            'avg_memory_mb': round(sum(memory_values) / len(memory_values) if memory_values else 0, 2),
            'avg_cpu_percent': round(sum(cpu_values) / len(cpu_values) if cpu_values else 0, 2),
            'avg_fps': round(sum(fps_values) / len(fps_values) if fps_values else 0, 2),
            'peak_fps': round(max(fps_values) if fps_values else 0, 2),
            'total_errors': len(self.processing_errors),
            'recovery_attempts': len(self.recovery_attempts),
            'recovery_success_rate': round(recovery_success_rate, 2),
            'memory_samples_count': len(self.memory_samples),
            'fps_samples_count': len(self.fps_samples)
        }


def create_test_video(duration_seconds=10, fps=30, resolution=(640, 480)):
    """Create a test video with specified properties"""
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


def test_performance_monitoring():
    """Test basic performance monitoring functionality"""
    print("Testing performance monitoring...")
    
    try:
        # Initialize performance monitor
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Simulate some processing activity
        for i in range(5):
            time.sleep(0.2)
            monitor.sample_performance()
            
            # Simulate some errors and recovery attempts
            if i == 2:
                monitor.record_error('test_error', 'Simulated error for testing', recovery_attempted=True)
                monitor.record_recovery_attempt('test_recovery', success=True)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Get performance summary
        perf_summary = monitor.get_performance_summary()
        
        # Verify results
        print(f"Performance summary: {perf_summary}")
        
        # Basic assertions
        assert perf_summary['duration'] > 0, "Processing duration should be positive"
        assert perf_summary['memory_samples_count'] > 0, "Should have memory samples"
        assert perf_summary['total_errors'] == 1, "Should have recorded 1 error"
        assert perf_summary['recovery_attempts'] == 1, "Should have recorded 1 recovery attempt"
        assert perf_summary['recovery_success_rate'] == 100.0, "Recovery should be 100% successful"
        
        print("✓ Performance monitoring test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_creation():
    """Test video creation functionality"""
    print("Testing video creation...")
    
    try:
        # Create test video
        video_path, temp_dir = create_test_video(duration_seconds=2, fps=30, resolution=(640, 480))
        
        # Verify video was created
        assert os.path.exists(video_path), "Video file should exist"
        
        # Verify video properties
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Video should be openable"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        assert frame_count == 60, f"Expected 60 frames, got {frame_count}"
        assert fps == 30.0, f"Expected 30 FPS, got {fps}"
        assert width == 640, f"Expected width 640, got {width}"
        assert height == 480, f"Expected height 480, got {height}"
        
        print(f"✓ Video created successfully: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            
        return True
        
    except Exception as e:
        print(f"✗ Video creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success1 = test_performance_monitoring()
    success2 = test_video_creation()
    
    if success1 and success2:
        print("\n✓ All performance tests passed!")
    else:
        print("\n✗ Some performance tests failed!")
        exit(1)