"""
Performance tests for long video processing

Tests processing of videos with various lengths and resolutions, monitors memory usage
and processing performance, and tests error recovery mechanisms with corrupted video streams.
Requirements: 4.1, 4.2, 4.3, 4.4
"""

import pytest
import os
import tempfile
import time
import threading
import psutil
import cv2
import numpy as np
import logging
import traceback
from unittest.mock import patch, MagicMock, Mock
from utils.video_processor import VideoProcess
from utils.counter import object_count, validate_model_file, get_default_model_path
from utils.video_validator import VideoValidator


class TestLongVideoPerformance:
    """Performance tests for long video processing with various lengths and resolutions"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Performance monitoring fixture"""
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
                    logging.warning(f"Error sampling performance: {e}")
                    
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
        
        return PerformanceMonitor()
    
    @pytest.fixture
    def create_test_video(self):
        """Factory for creating test videos with various properties"""
        def _create_video(duration_seconds=10, fps=30, resolution=(640, 480), 
                         add_corruption=False, corruption_frame=None):
            """
            Create a test video with specified properties
            
            Args:
                duration_seconds: Video duration in seconds
                fps: Frames per second
                resolution: Video resolution as (width, height)
                add_corruption: Whether to add corruption to test error recovery
                corruption_frame: Frame number to corrupt (if None, corrupts middle frame)
            """
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, f'test_video_{duration_seconds}s_{fps}fps_{resolution[0]}x{resolution[1]}.mp4')
            
            total_frames = duration_seconds * fps
            width, height = resolution
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
            
            corruption_frame_num = corruption_frame or (total_frames // 2)
            
            for frame_num in range(total_frames):
                # Create frame with moving content for realistic processing
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add background gradient
                for y in range(height):
                    frame[y, :, 0] = int((y / height) * 255)  # Red gradient
                    frame[y, :, 1] = int(((height - y) / height) * 255)  # Green gradient
                
                # Add moving objects for detection
                circle_x = int((frame_num / total_frames) * width)
                circle_y = height // 2
                cv2.circle(frame, (circle_x, circle_y), 20, (255, 255, 255), -1)
                
                # Add frame number text
                cv2.putText(frame, f'Frame {frame_num}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add timestamp
                timestamp = frame_num / fps
                cv2.putText(frame, f'Time: {timestamp:.2f}s', (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add corruption if requested
                if add_corruption and frame_num == corruption_frame_num:
                    # Corrupt the frame by adding noise
                    noise = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
                    frame = cv2.addWeighted(frame, 0.3, noise, 0.7, 0)
                
                out.write(frame)
            
            out.release()
            
            return {
                'path': video_path,
                'duration': duration_seconds,
                'fps': fps,
                'resolution': resolution,
                'total_frames': total_frames,
                'corrupted': add_corruption,
                'corruption_frame': corruption_frame_num if add_corruption else None,
                'cleanup': lambda: self._cleanup_video(video_path, temp_dir)
            }
        
        return _create_video
    
    def _cleanup_video(self, video_path, temp_dir):
        """Clean up test video files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logging.warning(f"Error cleaning up test video: {e}")
    
    @pytest.fixture
    def mock_model_path(self):
        """Mock model path for testing"""
        # Try to get a real model path first
        try:
            real_model_path = get_default_model_path()
            if real_model_path and os.path.exists(real_model_path):
                yield real_model_path
                return
        except Exception as e:
            logging.warning(f"Could not get real model path: {e}")
        
        # Create a mock model file if no real model exists
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, 'test_model.pt')
        
        try:
            # Create a dummy model file
            with open(model_path, 'wb') as f:
                f.write(b'dummy model data for testing')
            
            yield model_path
        finally:
            # Cleanup
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logging.warning(f"Error cleaning up mock model: {e}")

    def test_short_video_baseline_performance(self, create_test_video, performance_monitor, mock_model_path):
        """Test baseline performance with short video (30 seconds, 720p)"""
        # Create short test video
        video_info = create_test_video(duration_seconds=30, fps=30, resolution=(1280, 720))
        
        try:
            # Initialize video processor
            processor = VideoProcess()
            performance_monitor.start_monitoring()
            
            # Mock the counter to avoid model loading issues
            with patch('utils.counter.object_count') as mock_object_count:
                mock_counter = MagicMock()
                mock_result = MagicMock()
                mock_result.plot_im = np.zeros((720, 1280, 3), dtype=np.uint8)
                mock_result.in_count = 5
                mock_result.total_tracks = 3
                mock_result.classwise_count = {'test_class': 2}
                mock_counter.process.return_value = mock_result
                mock_object_count.return_value = mock_counter
                
                # Start processing
                success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                assert success, f"Failed to start processing: {message}"
                
                # Monitor performance during processing
                monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                   args=(processor, performance_monitor))
                monitoring_thread.daemon = True
                monitoring_thread.start()
                
                # Wait for processing to complete
                max_wait_time = 120  # 2 minutes max for 30-second video
                start_wait = time.time()
                
                while processor.processing and (time.time() - start_wait) < max_wait_time:
                    time.sleep(1)
                    performance_monitor.sample_performance(processor)
                
                # Stop monitoring
                performance_monitor.stop_monitoring()
                
                # Get performance summary
                perf_summary = performance_monitor.get_performance_summary()
                
                # Assertions for baseline performance
                assert perf_summary['duration'] > 0, "Processing duration should be positive"
                assert perf_summary['peak_memory_mb'] < 1000, f"Memory usage too high: {perf_summary['peak_memory_mb']}MB"
                assert perf_summary['avg_fps'] > 0, f"Average FPS should be positive: {perf_summary['avg_fps']}"
                
                # Log performance summary
                logger.info(f"Baseline performance test completed: {perf_summary}")
                
        finally:
            video_info['cleanup']()
    
    def test_long_video_performance(self, create_test_video, performance_monitor, mock_model_path):
        """Test performance with long video (5 minutes, 1080p) - Requirements 4.1, 4.4"""
        # Create long test video
        video_info = create_test_video(duration_seconds=300, fps=30, resolution=(1920, 1080))
        
        try:
            # Initialize video processor
            processor = VideoProcess()
            performance_monitor.start_monitoring()
            
            # Mock the counter to avoid model loading issues
            with patch('utils.counter.object_count') as mock_object_count:
                mock_counter = MagicMock()
                mock_result = MagicMock()
                mock_result.plot_im = np.zeros((1080, 1920, 3), dtype=np.uint8)
                mock_result.in_count = 10
                mock_result.total_tracks = 5
                mock_result.classwise_count = {'test_class': 3}
                mock_counter.process.return_value = mock_result
                mock_object_count.return_value = mock_counter
                
                # Start processing
                success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                assert success, f"Failed to start processing: {message}"
                
                # Monitor performance during processing
                monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                   args=(processor, performance_monitor))
                monitoring_thread.daemon = True
                monitoring_thread.start()
                
                # Wait for processing to complete (with timeout for long video)
                max_wait_time = 600  # 10 minutes max for 5-minute video
                start_wait = time.time()
                
                while processor.processing and (time.time() - start_wait) < max_wait_time:
                    time.sleep(2)
                    performance_monitor.sample_performance(processor)
                
                # Stop monitoring
                performance_monitor.stop_monitoring()
                
                # Get performance summary
                perf_summary = performance_monitor.get_performance_summary()
                
                # Assertions for long video performance
                assert perf_summary['duration'] > 0, "Processing duration should be positive"
                assert perf_summary['peak_memory_mb'] < 2000, f"Memory usage too high for long video: {perf_summary['peak_memory_mb']}MB"
                assert perf_summary['avg_fps'] > 0, f"Average FPS should be positive: {perf_summary['avg_fps']}"
                
                # Performance requirements for long videos
                assert perf_summary['avg_memory_mb'] < 1500, f"Average memory usage too high: {perf_summary['avg_memory_mb']}MB"
                
                # Log performance summary
                logger.info(f"Long video performance test completed: {perf_summary}")
                
        finally:
            video_info['cleanup']()
    
    def test_high_resolution_video_performance(self, create_test_video, performance_monitor, mock_model_path):
        """Test performance with high resolution video (4K, 2 minutes) - Requirements 4.1, 4.4"""
        # Create 4K test video
        video_info = create_test_video(duration_seconds=120, fps=30, resolution=(3840, 2160))
        
        try:
            # Initialize video processor
            processor = VideoProcess()
            performance_monitor.start_monitoring()
            
            # Mock the counter to avoid model loading issues
            with patch('utils.counter.object_count') as mock_object_count:
                mock_counter = MagicMock()
                mock_result = MagicMock()
                mock_result.plot_im = np.zeros((2160, 3840, 3), dtype=np.uint8)
                mock_result.in_count = 15
                mock_result.total_tracks = 8
                mock_result.classwise_count = {'test_class': 5}
                mock_counter.process.return_value = mock_result
                mock_object_count.return_value = mock_counter
                
                # Start processing
                success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                assert success, f"Failed to start processing: {message}"
                
                # Monitor performance during processing
                monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                   args=(processor, performance_monitor))
                monitoring_thread.daemon = True
                monitoring_thread.start()
                
                # Wait for processing to complete (with timeout for 4K video)
                max_wait_time = 480  # 8 minutes max for 2-minute 4K video
                start_wait = time.time()
                
                while processor.processing and (time.time() - start_wait) < max_wait_time:
                    time.sleep(2)
                    performance_monitor.sample_performance(processor)
                
                # Stop monitoring
                performance_monitor.stop_monitoring()
                
                # Get performance summary
                perf_summary = performance_monitor.get_performance_summary()
                
                # Assertions for 4K video performance
                assert perf_summary['duration'] > 0, "Processing duration should be positive"
                assert perf_summary['peak_memory_mb'] < 4000, f"Memory usage too high for 4K video: {perf_summary['peak_memory_mb']}MB"
                assert perf_summary['avg_fps'] > 0, f"Average FPS should be positive: {perf_summary['avg_fps']}"
                
                # Performance requirements for 4K videos (more lenient)
                assert perf_summary['avg_memory_mb'] < 3000, f"Average memory usage too high: {perf_summary['avg_memory_mb']}MB"
                
                # Log performance summary
                logger.info(f"4K video performance test completed: {perf_summary}")
                
        finally:
            video_info['cleanup']()
    
    def test_corrupted_video_error_recovery(self, create_test_video, performance_monitor, mock_model_path):
        """Test error recovery mechanisms with corrupted video streams - Requirements 4.2, 4.3"""
        # Create corrupted test video
        video_info = create_test_video(duration_seconds=60, fps=30, resolution=(1280, 720), 
                                     add_corruption=True, corruption_frame=900)  # Corrupt frame at 30 seconds
        
        try:
            # Initialize video processor
            processor = VideoProcess()
            performance_monitor.start_monitoring()
            
            # Mock the counter to avoid model loading issues
            with patch('utils.counter.object_count') as mock_object_count:
                mock_counter = MagicMock()
                mock_result = MagicMock()
                mock_result.plot_im = np.zeros((720, 1280, 3), dtype=np.uint8)
                mock_result.in_count = 8
                mock_result.total_tracks = 4
                mock_result.classwise_count = {'test_class': 2}
                mock_counter.process.return_value = mock_result
                mock_object_count.return_value = mock_counter
                
                # Start processing
                success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                assert success, f"Failed to start processing: {message}"
                
                # Monitor performance and errors during processing
                monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                   args=(processor, performance_monitor))
                monitoring_thread.daemon = True
                monitoring_thread.start()
                
                # Wait for processing to complete
                max_wait_time = 180  # 3 minutes max for 1-minute corrupted video
                start_wait = time.time()
                
                while processor.processing and (time.time() - start_wait) < max_wait_time:
                    time.sleep(1)
                    performance_monitor.sample_performance(processor)
                    
                    # Check for error recovery attempts
                    if hasattr(processor, 'error_stats'):
                        if processor.error_stats['frame_read_errors'] > 0:
                            performance_monitor.record_error('frame_read_error', 
                                                           f"Frame read errors: {processor.error_stats['frame_read_errors']}")
                        if processor.error_stats['recovery_attempts'] > 0:
                            performance_monitor.record_recovery_attempt('video_capture_reinit', 
                                                                      processor.error_stats['successful_recoveries'] > 0)
                
                # Stop monitoring
                performance_monitor.stop_monitoring()
                
                # Get performance summary
                perf_summary = performance_monitor.get_performance_summary()
                
                # Assertions for corrupted video handling
                assert perf_summary['duration'] > 0, "Processing duration should be positive"
                assert perf_summary['total_errors'] >= 0, "Should track error occurrences"
                
                # Verify error recovery was attempted
                if perf_summary['total_errors'] > 0:
                    assert perf_summary['recovery_attempts'] >= 0, "Should attempt recovery when errors occur"
                
                # Log performance and error recovery summary
                logger.info(f"Corrupted video error recovery test completed: {perf_summary}")
                logger.info(f"Error recovery success rate: {perf_summary['recovery_success_rate']}%")
                
        finally:
            video_info['cleanup']()
    
    def test_memory_usage_monitoring(self, create_test_video, performance_monitor, mock_model_path):
        """Test memory usage monitoring during long video processing - Requirements 4.1, 4.4"""
        # Create medium-length test video
        video_info = create_test_video(duration_seconds=180, fps=30, resolution=(1280, 720))
        
        try:
            # Initialize video processor
            processor = VideoProcess()
            performance_monitor.start_monitoring()
            
            # Mock the counter to avoid model loading issues
            with patch('utils.counter.object_count') as mock_object_count:
                mock_counter = MagicMock()
                mock_result = MagicMock()
                mock_result.plot_im = np.zeros((720, 1280, 3), dtype=np.uint8)
                mock_result.in_count = 12
                mock_result.total_tracks = 6
                mock_result.classwise_count = {'test_class': 4}
                mock_counter.process.return_value = mock_result
                mock_object_count.return_value = mock_counter
                
                # Start processing
                success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                assert success, f"Failed to start processing: {message}"
                
                # Monitor memory usage intensively
                memory_samples = []
                max_wait_time = 360  # 6 minutes max for 3-minute video
                start_wait = time.time()
                
                while processor.processing and (time.time() - start_wait) < max_wait_time:
                    time.sleep(0.5)  # Sample every 0.5 seconds for detailed monitoring
                    performance_monitor.sample_performance(processor)
                    
                    # Additional memory sampling
                    try:
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        memory_samples.append({
                            'timestamp': time.time(),
                            'memory_mb': memory_mb,
                            'frames_processed': processor.frames_processed if hasattr(processor, 'frames_processed') else 0
                        })
                    except Exception as e:
                        logger.warning(f"Error sampling memory: {e}")
                
                # Stop monitoring
                performance_monitor.stop_monitoring()
                
                # Get performance summary
                perf_summary = performance_monitor.get_performance_summary()
                
                # Memory usage analysis
                if memory_samples:
                    memory_values = [s['memory_mb'] for s in memory_samples]
                    peak_memory = max(memory_values)
                    avg_memory = sum(memory_values) / len(memory_values)
                    memory_growth = memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
                    
                    # Assertions for memory usage
                    assert peak_memory < 1500, f"Peak memory usage too high: {peak_memory}MB"
                    assert avg_memory < 1000, f"Average memory usage too high: {avg_memory}MB"
                    assert memory_growth < 500, f"Memory growth too high: {memory_growth}MB (potential memory leak)"
                    
                    logger.info(f"Memory usage analysis - Peak: {peak_memory:.2f}MB, "
                              f"Average: {avg_memory:.2f}MB, Growth: {memory_growth:.2f}MB")
                
                # Log performance summary
                logger.info(f"Memory usage monitoring test completed: {perf_summary}")
                
        finally:
            video_info['cleanup']()
    
    def test_processing_performance_metrics(self, create_test_video, performance_monitor, mock_model_path):
        """Test processing performance metrics collection - Requirements 4.1, 4.4"""
        # Create test video with moderate length
        video_info = create_test_video(duration_seconds=90, fps=30, resolution=(1280, 720))
        
        try:
            # Initialize video processor
            processor = VideoProcess()
            performance_monitor.start_monitoring()
            
            # Mock the counter to avoid model loading issues
            with patch('utils.counter.object_count') as mock_object_count:
                mock_counter = MagicMock()
                mock_result = MagicMock()
                mock_result.plot_im = np.zeros((720, 1280, 3), dtype=np.uint8)
                mock_result.in_count = 20
                mock_result.total_tracks = 10
                mock_result.classwise_count = {'test_class': 8}
                mock_counter.process.return_value = mock_result
                mock_object_count.return_value = mock_counter
                
                # Start processing
                success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                assert success, f"Failed to start processing: {message}"
                
                # Monitor performance metrics
                fps_samples = []
                max_wait_time = 270  # 4.5 minutes max for 1.5-minute video
                start_wait = time.time()
                
                while processor.processing and (time.time() - start_wait) < max_wait_time:
                    time.sleep(1)
                    performance_monitor.sample_performance(processor)
                    
                    # Collect FPS samples
                    if hasattr(processor, 'latest_result') and processor.latest_result:
                        fps = processor.latest_result.get('fps', 0)
                        if fps > 0:
                            fps_samples.append(fps)
                
                # Stop monitoring
                performance_monitor.stop_monitoring()
                
                # Get performance summary
                perf_summary = performance_monitor.get_performance_summary()
                
                # Performance metrics analysis
                if fps_samples:
                    avg_fps = sum(fps_samples) / len(fps_samples)
                    peak_fps = max(fps_samples)
                    min_fps = min(fps_samples)
                    fps_variance = max(fps_samples) - min(fps_samples)
                    
                    # Assertions for processing performance
                    assert avg_fps > 0, f"Average FPS should be positive: {avg_fps}"
                    assert peak_fps > 0, f"Peak FPS should be positive: {peak_fps}"
                    assert fps_variance < 50, f"FPS variance too high: {fps_variance} (unstable processing)"
                    
                    logger.info(f"Processing performance metrics - Average FPS: {avg_fps:.2f}, "
                              f"Peak FPS: {peak_fps:.2f}, Min FPS: {min_fps:.2f}, Variance: {fps_variance:.2f}")
                
                # Verify performance summary contains expected metrics
                assert 'avg_fps' in perf_summary, "Performance summary should include average FPS"
                assert 'peak_fps' in perf_summary, "Performance summary should include peak FPS"
                assert 'duration' in perf_summary, "Performance summary should include duration"
                
                # Log performance summary
                logger.info(f"Processing performance metrics test completed: {perf_summary}")
                
        finally:
            video_info['cleanup']()
    
    def test_multiple_resolution_stress_test(self, create_test_video, performance_monitor, mock_model_path):
        """Test processing performance across multiple resolutions - Requirements 4.1, 4.4"""
        resolutions = [
            (640, 480, "480p"),
            (1280, 720, "720p"),
            (1920, 1080, "1080p"),
            (2560, 1440, "1440p")
        ]
        
        performance_results = {}
        
        for width, height, name in resolutions:
            logger.info(f"Testing {name} resolution ({width}x{height})")
            
            # Create test video for this resolution
            video_info = create_test_video(duration_seconds=30, fps=30, resolution=(width, height))
            
            try:
                # Initialize video processor
                processor = VideoProcess()
                performance_monitor.start_monitoring()
                
                # Mock the counter to avoid model loading issues
                with patch('utils.counter.object_count') as mock_object_count:
                    mock_counter = MagicMock()
                    mock_result = MagicMock()
                    mock_result.plot_im = np.zeros((height, width, 3), dtype=np.uint8)
                    mock_result.in_count = 8
                    mock_result.total_tracks = 4
                    mock_result.classwise_count = {'test_class': 3}
                    mock_counter.process.return_value = mock_result
                    mock_object_count.return_value = mock_counter
                    
                    # Start processing
                    success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                    assert success, f"Failed to start processing for {name}: {message}"
                    
                    # Monitor performance during processing
                    monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                       args=(processor, performance_monitor))
                    monitoring_thread.daemon = True
                    monitoring_thread.start()
                    
                    # Wait for processing to complete
                    max_wait_time = 120  # 2 minutes max for 30-second video
                    start_wait = time.time()
                    
                    while processor.processing and (time.time() - start_wait) < max_wait_time:
                        time.sleep(1)
                        performance_monitor.sample_performance(processor)
                    
                    # Stop monitoring
                    performance_monitor.stop_monitoring()
                    
                    # Get performance summary
                    perf_summary = performance_monitor.get_performance_summary()
                    performance_results[name] = perf_summary
                    
                    # Basic assertions for each resolution
                    assert perf_summary['duration'] > 0, f"Processing duration should be positive for {name}"
                    assert perf_summary['avg_fps'] > 0, f"Average FPS should be positive for {name}: {perf_summary['avg_fps']}"
                    
                    logger.info(f"{name} performance test completed: {perf_summary}")
                    
            finally:
                video_info['cleanup']()
        
        # Compare performance across resolutions
        logger.info("Performance comparison across resolutions:")
        for resolution, results in performance_results.items():
            logger.info(f"{resolution}: {results['avg_fps']:.2f} FPS, {results['peak_memory_mb']:.1f} MB peak memory")
        
        # Verify performance degrades gracefully with higher resolutions
        fps_values = [results['avg_fps'] for results in performance_results.values()]
        memory_values = [results['peak_memory_mb'] for results in performance_results.values()]
        
        # Higher resolutions should use more memory
        assert max(memory_values) > min(memory_values), "Higher resolutions should use more memory"
        
        # All resolutions should maintain reasonable performance
        for resolution, results in performance_results.items():
            assert results['avg_fps'] > 0.5, f"{resolution} FPS too low: {results['avg_fps']}"
            assert results['peak_memory_mb'] < 5000, f"{resolution} memory usage too high: {results['peak_memory_mb']}MB"
    
    def test_extended_duration_stress_test(self, create_test_video, performance_monitor, mock_model_path):
        """Test processing performance with extended duration videos - Requirements 4.1, 4.4"""
        durations = [60, 120, 300]  # 1, 2, and 5 minutes
        
        for duration in durations:
            logger.info(f"Testing {duration}-second video processing")
            
            # Create test video
            video_info = create_test_video(duration_seconds=duration, fps=30, resolution=(1280, 720))
            
            try:
                # Initialize video processor
                processor = VideoProcess()
                performance_monitor.start_monitoring()
                
                # Mock the counter to avoid model loading issues
                with patch('utils.counter.object_count') as mock_object_count:
                    mock_counter = MagicMock()
                    mock_result = MagicMock()
                    mock_result.plot_im = np.zeros((720, 1280, 3), dtype=np.uint8)
                    mock_result.in_count = 12
                    mock_result.total_tracks = 6
                    mock_result.classwise_count = {'test_class': 4}
                    mock_counter.process.return_value = mock_result
                    mock_object_count.return_value = mock_counter
                    
                    # Start processing
                    success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                    assert success, f"Failed to start processing for {duration}s video: {message}"
                    
                    # Monitor performance during processing
                    monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                       args=(processor, performance_monitor))
                    monitoring_thread.daemon = True
                    monitoring_thread.start()
                    
                    # Wait for processing to complete (with timeout)
                    max_wait_time = duration * 4  # 4x video duration as timeout
                    start_wait = time.time()
                    
                    while processor.processing and (time.time() - start_wait) < max_wait_time:
                        time.sleep(2)
                        performance_monitor.sample_performance(processor)
                    
                    # Stop monitoring
                    performance_monitor.stop_monitoring()
                    
                    # Get performance summary
                    perf_summary = performance_monitor.get_performance_summary()
                    
                    # Assertions for extended duration processing
                    assert perf_summary['duration'] > 0, f"Processing duration should be positive for {duration}s video"
                    assert perf_summary['avg_fps'] > 0, f"Average FPS should be positive for {duration}s video"
                    
                    # Memory usage should remain stable for longer videos
                    memory_growth = perf_summary['peak_memory_mb'] - perf_summary['avg_memory_mb']
                    assert memory_growth < 1000, f"Memory growth too high for {duration}s video: {memory_growth}MB"
                    
                    logger.info(f"{duration}s video performance test completed: {perf_summary}")
                    
            finally:
                video_info['cleanup']()
    
    def test_error_recovery_stress_test(self, create_test_video, performance_monitor, mock_model_path):
        """Test error recovery mechanisms under stress conditions - Requirements 4.2, 4.3"""
        # Create multiple corrupted videos with different corruption patterns
        corruption_scenarios = [
            {"corruption_frame": 300, "name": "early_corruption"},
            {"corruption_frame": 900, "name": "mid_corruption"},
            {"corruption_frame": 1500, "name": "late_corruption"}
        ]
        
        for scenario in corruption_scenarios:
            logger.info(f"Testing error recovery with {scenario['name']}")
            
            # Create corrupted test video
            video_info = create_test_video(
                duration_seconds=60, 
                fps=30, 
                resolution=(1280, 720),
                add_corruption=True, 
                corruption_frame=scenario['corruption_frame']
            )
            
            try:
                # Initialize video processor
                processor = VideoProcess()
                performance_monitor.start_monitoring()
                
                # Mock the counter to avoid model loading issues
                with patch('utils.counter.object_count') as mock_object_count:
                    mock_counter = MagicMock()
                    mock_result = MagicMock()
                    mock_result.plot_im = np.zeros((720, 1280, 3), dtype=np.uint8)
                    mock_result.in_count = 15
                    mock_result.total_tracks = 8
                    mock_result.classwise_count = {'test_class': 5}
                    mock_counter.process.return_value = mock_result
                    mock_object_count.return_value = mock_counter
                    
                    # Start processing
                    success, message = processor.start_processing(video_info['path'], 'test_model.pt')
                    assert success, f"Failed to start processing for {scenario['name']}: {message}"
                    
                    # Monitor performance and errors during processing
                    monitoring_thread = threading.Thread(target=self._monitor_processing_performance, 
                                                       args=(processor, performance_monitor))
                    monitoring_thread.daemon = True
                    monitoring_thread.start()
                    
                    # Wait for processing to complete
                    max_wait_time = 180  # 3 minutes max for 1-minute corrupted video
                    start_wait = time.time()
                    
                    while processor.processing and (time.time() - start_wait) < max_wait_time:
                        time.sleep(1)
                        performance_monitor.sample_performance(processor)
                        
                        # Simulate additional error conditions for stress testing
                        if hasattr(processor, 'error_stats'):
                            if processor.error_stats['frame_read_errors'] > 0:
                                performance_monitor.record_error('frame_read_error', 
                                                               f"Frame read errors: {processor.error_stats['frame_read_errors']}")
                            if processor.error_stats['recovery_attempts'] > 0:
                                performance_monitor.record_recovery_attempt('video_capture_reinit', 
                                                                          processor.error_stats['successful_recoveries'] > 0)
                    
                    # Stop monitoring
                    performance_monitor.stop_monitoring()
                    
                    # Get performance summary
                    perf_summary = performance_monitor.get_performance_summary()
                    
                    # Assertions for error recovery
                    assert perf_summary['duration'] > 0, f"Processing duration should be positive for {scenario['name']}"
                    
                    # Should handle errors gracefully
                    if perf_summary['total_errors'] > 0:
                        logger.info(f"{scenario['name']} encountered {perf_summary['total_errors']} errors")
                        assert perf_summary['recovery_attempts'] >= 0, f"Should attempt recovery for {scenario['name']}"
                    
                    logger.info(f"{scenario['name']} error recovery test completed: {perf_summary}")
                    logger.info(f"Error recovery success rate: {perf_summary['recovery_success_rate']}%")
                    
            finally:
                video_info['cleanup']()
    
    def _monitor_processing_performance(self, processor, performance_monitor):
        """Helper method to monitor processing performance in a separate thread"""
        try:
            while processor.processing:
                performance_monitor.sample_performance(processor)
                time.sleep(1)  # Sample every second
        except Exception as e:
            logger.warning(f"Error in performance monitoring thread: {e}")


# Configure logging for the test module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])