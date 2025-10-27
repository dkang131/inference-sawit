import cv2
import threading
import time
import queue
import traceback
import numpy as np
import logging
import os
import sys

# Configure logger with handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class VideoProcess:
    def __init__(self):
        self.processing = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=50)  # Increased queue size for longer videos
        self.latest_result = None
        self.last_valid_result = None
        self.counter = None
        self.cap = None
        self.video_writer = None  # Video writer for saving output
        self.output_video_path = None  # Path to save output video
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.process_thread = None
        self.frame_count = 0
        self.end_of_video = False
        self.processing_complete = False
        self.total_frames = 0
        self.frames_processed = 0
        self.video_source = None  # Store video source for potential reinitialization
        self.model_name = None  # Store model name for database saving
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 30.0
        self.consecutive_read_failures = 0
        self.max_consecutive_read_failures = 300  # Increased for longer videos
        self.start_time = None  # Track processing start time
        self.last_progress_log = 0  # Track when we last logged progress
        
        # Enhanced end-of-video detection attributes
        self.video_completion_flags = {
            'frame_position_reached': False,
            'read_failures_exceeded': False,
            'total_frames_processed': False,
            'capture_thread_finished': False,
            'processing_thread_finished': False
        }
        self.video_completion_status = {
            'completion_method': None,
            'completion_timestamp': None,
            'final_frame_position': 0,
            'final_frames_processed': 0,
            'completion_reason': None
        }
        
        # Error tracking and recovery attributes
        self.error_stats = {
            'frame_read_errors': 0,
            'frame_processing_errors': 0,
            'video_write_errors': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'cuda_errors': 0,  # Track CUDA-specific errors
            'recent_cuda_errors': 0  # Track recent CUDA errors to prevent premature completion
        }
        self.last_successful_frame_time = time.time()
        self.processing_health_check_interval = 30.0  # Check processing health every 30 seconds
        
        # Enhanced progress tracking attributes
        self.progress_tracking = {
            'start_time': None,
            'last_progress_update': 0,
            'progress_update_interval': 5.0,  # Update progress every 5 seconds
            'frames_per_second_history': [],
            'estimated_time_remaining': 0,
            'processing_stages': {
                'initialization': {'start': None, 'end': None, 'duration': 0},
                'capture': {'start': None, 'end': None, 'duration': 0},
                'processing': {'start': None, 'end': None, 'duration': 0},
                'cleanup': {'start': None, 'end': None, 'duration': 0}
            },
            'performance_metrics': {
                'avg_frame_processing_time': 0,
                'avg_capture_fps': 0,
                'avg_processing_fps': 0,
                'peak_processing_fps': 0,
                'memory_usage_mb': 0
            }
        }
        
        # Enhanced logging configuration
        self.logging_config = {
            'progress_log_interval': 25,  # Log progress every 25 frames
            'performance_log_interval': 100,  # Log performance every 100 frames
            'debug_log_interval': 10,  # Debug logs every 10 frames
            'detailed_progress_enabled': True,
            'performance_tracking_enabled': True
        }
        
        # Manual stop tracking
        self.manually_stopped = False
        self.stop_requested_time = None
        
        # Heartbeat tracking to detect silent failures
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 10.0  # Update heartbeat every 10 seconds
        self.processing_stalled = False
    
    def capture_frames(self, video_path):
        logger.info(f"[CAPTURE] Starting capture_frames for video: {video_path}")
        
        frame_index = 0
        retry_count = 0
        max_retries = 5  # Increased retries for longer videos
        frame_read_retry_count = 0
        max_frame_read_retries = 3  # Maximum retries for individual frame reads
        
        # Validate video capture is available
        if not self.cap:
            logger.error("[CAPTURE] Video capture is None, cannot proceed")
            return
        
        if not self.cap.isOpened():
            logger.error("[CAPTURE] Video capture is not opened, cannot proceed")
            return
        
        # Get video FPS for proper timing
        if self.cap:
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)  # pyright: ignore[reportAttributeAccessIssue]
            if video_fps <= 0:
                video_fps = self.video_fps  # Use default or previously detected FPS
        else:
            video_fps = self.video_fps
            
        frame_delay = 1.0 / video_fps
        logger.info(f"[CAPTURE] Capture timing - FPS: {video_fps:.2f}, frame delay: {frame_delay:.4f}s")
        logger.info(f"[CAPTURE] Video properties - Total frames: {self.total_frames}, Stop event set: {self.stop_event.is_set()}")

        while (not self.stop_event.is_set() and 
               self.cap and 
               self.cap.isOpened() and 
               not self._update_video_completion_status()):
            
            # Check if we've reached the end of the video explicitly
            if self.total_frames > 0:
                current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                # Add a small buffer to account for potential rounding issues
                if current_frame_pos >= self.total_frames - 5:
                    logger.info(f"[CAPTURE] End of video reached - processed {frame_index} frames out of {self.total_frames}")
                    logger.info(f"[CAPTURE] Current frame position: {current_frame_pos}, Total frames: {self.total_frames}")
                    break
            
            # Log frame reading attempt
            logger.debug(f"[CAPTURE] Attempting to read frame {frame_index}")
            ret, frame = self.cap.read()
            
            if not ret:
                self.consecutive_read_failures += 1
                self.error_stats['frame_read_errors'] += 1
                frame_read_retry_count += 1
                logger.warning(f"[CAPTURE] Failed to read frame {frame_index}. "
                             f"Consecutive failures: {self.consecutive_read_failures}, "
                             f"Frame retry: {frame_read_retry_count}, "
                             f"Total read errors: {self.error_stats['frame_read_errors']}")
                
                # Log current position for debugging
                if self.cap:
                    current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                    logger.debug(f"[CAPTURE] Current position after failed read: {current_pos}")
                    
                    # Log detailed error context for debugging
                    if frame_read_retry_count == 1:  # Only on first retry to avoid spam
                        logger.debug(f"[CAPTURE] Error context - Frame index: {frame_index}, "
                                   f"Position: {current_pos}, Total frames: {self.total_frames}, "
                                   f"Video opened: {self.cap.isOpened()}")
                
                # Enhanced retry logic for individual frame reads
                if frame_read_retry_count <= max_frame_read_retries:
                    logger.info(f"[CAPTURE] Retrying frame read {frame_read_retry_count}/{max_frame_read_retries}")
                    # Try to seek back one frame and read again
                    try:
                        if self.cap and current_pos > 0:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))  # pyright: ignore[reportAttributeAccessIssue]
                            logger.debug(f"[CAPTURE] Seeked back to frame {max(0, current_pos - 1)} for retry")
                    except Exception as e:
                        logger.warning(f"[CAPTURE] Failed to seek back for retry: {e}")
                    
                    time.sleep(0.05)  # Brief pause before retry
                    continue
                
                # Reset frame retry counter after exhausting retries
                frame_read_retry_count = 0
                
                # Try to reinitialize video capture if failures are frequent
                if self.consecutive_read_failures % 50 == 0 and retry_count < max_retries:
                    logger.info(f"[CAPTURE] Attempting to reinitialize video capture (retry {retry_count + 1}) after {self.consecutive_read_failures} consecutive failures")
                    if self._reinitialize_video_capture_with_recovery(video_path, frame_index):
                        retry_count += 1
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                    else:
                        logger.error(f"[CAPTURE] Failed to reinitialize video capture, continuing with current stream")
                
                # Add exponential backoff for persistent failures
                if self.consecutive_read_failures > 100:
                    backoff_time = min(0.1 * (self.consecutive_read_failures // 100), 1.0)
                    logger.debug(f"[CAPTURE] Applying backoff delay: {backoff_time:.3f}s")
                    time.sleep(backoff_time)
                else:
                    time.sleep(0.01)  # Standard delay to prevent busy waiting
                continue
            
            # Reset failure counters on successful read
            if self.consecutive_read_failures > 0:
                logger.debug(f"[CAPTURE] Resetting consecutive read failures from {self.consecutive_read_failures} to 0 after successful read")
            self.consecutive_read_failures = 0
            retry_count = 0  # Reset retry count on successful read
            frame_read_retry_count = 0  # Reset frame retry counter on successful read
            
            # Update processing health
            self._update_processing_health()
            
            # Log successful frame read
            logger.debug(f"[CAPTURE] Successfully read frame {frame_index}")
            
            # Add frame to queue with index
            try:
                # If queue is full, remove oldest frame and add new one
                if self.frame_queue.full():
                    try:
                        dropped_frame = self.frame_queue.get_nowait()
                        logger.debug(f"[CAPTURE] Dropped old frame {dropped_frame[0]} to make room for new frame {frame_index}")
                    except queue.Empty:
                        pass
                
                self.frame_queue.put((frame_index, frame), timeout=2)  # Increased timeout
                frame_index += 1
                
                # Enhanced progress logging with detailed tracking
                if frame_index % self.logging_config['debug_log_interval'] == 0:
                    logger.debug(f"[CAPTURE] Frame {frame_index} captured successfully")
                
                # Log comprehensive progress at regular intervals
                if frame_index % self.logging_config['progress_log_interval'] == 0:
                    self._log_detailed_progress()
                    
                    # Log current position and check for completion
                    if self.cap:
                        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                        logger.debug(f"[CAPTURE] Current position: {current_pos}")
                        
                        # Check if we're approaching the end of the video
                        if current_pos >= self.total_frames - 10:
                            logger.info(f"[CAPTURE] Approaching end of video - Position: {current_pos}, Total: {self.total_frames}")
                            # Update completion status to check all methods
                            if self._update_video_completion_status():
                                logger.info("[CAPTURE] Video completion detected during progress check")
                                break
                    
                # Add timing control to match video FPS
                time.sleep(frame_delay)
                
            except queue.Full:
                logger.warning(f"[CAPTURE] Frame queue is full, dropping frame {frame_index}")
                continue
            
        logger.info(f"[CAPTURE] Capture thread finished. Total frames captured: {frame_index}")
        logger.info(f"[CAPTURE] Consecutive read failures: {self.consecutive_read_failures}")
        
        # Mark capture thread as finished and update completion status
        self.video_completion_flags['capture_thread_finished'] = True
        self._update_video_completion_status()
        
        logger.info(f"[CAPTURE] Final completion status: {self.get_completion_status()}")

    def _reinitialize_video_capture(self, video_path):
        """Reinitialize video capture in case of stream corruption"""
        try:
            logger.info(f"[REINIT] Starting video capture reinitialization for: {video_path}")
            
            if self.cap:
                logger.info("[REINIT] Releasing current video capture")
                self.cap.release()
            
            logger.info("[REINIT] Creating new VideoCapture instance")
            self.cap = cv2.VideoCapture(video_path)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            
            # Seek to the current frame position
            if self.frames_processed > 0:
                # Try to seek to the exact frame
                target_frame = self.frames_processed
                logger.info(f"[REINIT] Seeking to frame position: {target_frame}")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)  # pyright: ignore[reportAttributeAccessIssue]
                
                # Verify we're at the correct position
                actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                logger.info(f"[REINIT] Seeking result - Requested: {target_frame}, Actual: {actual_frame}")
                
                # If seeking didn't work properly, log a warning
                if abs(actual_frame - target_frame) > 5:
                    logger.warning(f"[REINIT] Frame seeking imprecise. Requested: {target_frame}, Got: {actual_frame}")
            
            if self.cap.isOpened():
                logger.info("[REINIT] Video capture reinitialized successfully")
                # Log current properties after reinitialization
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                logger.info(f"[REINIT] Current position after reinitialization: {current_pos}")
                return True
            else:
                logger.error("[REINIT] Failed to reinitialize video capture")
                return False
        except Exception as e:
            logger.error(f"[REINIT] Error reinitializing video capture: {e}")
            traceback.print_exc()
            return False

    def _reinitialize_video_capture_with_recovery(self, video_path, current_frame_index):
        """Enhanced video capture reinitialization with multiple recovery strategies"""
        try:
            logger.info(f"[REINIT_ENHANCED] Starting enhanced video capture reinitialization for: {video_path}")
            logger.info(f"[REINIT_ENHANCED] Current frame index: {current_frame_index}, frames processed: {self.frames_processed}")
            
            # Store current state for recovery
            original_consecutive_failures = self.consecutive_read_failures
            
            # Strategy 1: Standard reinitialization
            if self._reinitialize_video_capture(video_path):
                logger.info("[REINIT_ENHANCED] Standard reinitialization successful")
                self.consecutive_read_failures = 0  # Reset failure counter on success
                return True
            
            # Strategy 2: Try different backend
            logger.info("[REINIT_ENHANCED] Trying reinitialization with different backend")
            try:
                if self.cap:
                    self.cap.release()
                
                # Try with CAP_FFMPEG backend
                self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                
                if self.cap.isOpened():
                    # Seek to current position
                    target_frame = max(0, current_frame_index - 10)  # Go back a few frames for safety
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)  # pyright: ignore[reportAttributeAccessIssue]
                    
                    actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                    logger.info(f"[REINIT_ENHANCED] FFMPEG backend successful. Position: {actual_frame}")
                    self.consecutive_read_failures = 0
                    return True
            except Exception as e:
                logger.warning(f"[REINIT_ENHANCED] FFMPEG backend failed: {e}")
            
            # Strategy 3: Try seeking to a different position
            logger.info("[REINIT_ENHANCED] Trying reinitialization with position adjustment")
            try:
                if self.cap:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(video_path)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                
                if self.cap.isOpened():
                    # Try seeking to a position further back
                    fallback_positions = [
                        max(0, current_frame_index - 50),
                        max(0, current_frame_index - 100),
                        max(0, current_frame_index - 200),
                        0  # Last resort: start from beginning
                    ]
                    
                    for fallback_pos in fallback_positions:
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_pos)  # pyright: ignore[reportAttributeAccessIssue]
                            actual_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
                            
                            # Test if we can read a frame from this position
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                logger.info(f"[REINIT_ENHANCED] Successfully recovered at position {actual_pos}")
                                # Seek back to the position we want to continue from
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_index - 10))  # pyright: ignore[reportAttributeAccessIssue]
                                self.consecutive_read_failures = 0
                                return True
                        except Exception as e:
                            logger.debug(f"[REINIT_ENHANCED] Fallback position {fallback_pos} failed: {e}")
                            continue
            except Exception as e:
                logger.warning(f"[REINIT_ENHANCED] Position adjustment strategy failed: {e}")
            
            # Strategy 4: Partial reset with reduced failure threshold
            logger.warning("[REINIT_ENHANCED] All recovery strategies failed, applying partial reset")
            self.consecutive_read_failures = min(original_consecutive_failures, self.max_consecutive_read_failures // 2)
            
            return False
            
        except Exception as e:
            logger.error(f"[REINIT_ENHANCED] Critical error during enhanced reinitialization: {e}")
            traceback.print_exc()
            return False

    def _initialize_video_writer(self, video_source, model_name=None):
        """
        Initialize video writer with proper codec and parameters.
        Handles initialization failures with fallback codecs and ensures proper cleanup.
        
        Args:
            video_source (str): Path to input video file
            model_name (str, optional): Name of model for output filename
        """
        try:
            logger.info("[VIDEO_WRITER] Starting video writer initialization")
            
            # Create outputs directory if it doesn't exist
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
                logger.info(f"[VIDEO_WRITER] Created outputs directory: {outputs_dir}")
            
            # Generate output filename based on input filename and model
            input_filename = os.path.basename(video_source)
            name, ext = os.path.splitext(input_filename)
            
            # Sanitize model name for filename
            safe_model_name = (model_name or 'default').replace('.', '_').replace('/', '_').replace('\\', '_')
            output_filename = f"{name}_processed_{safe_model_name}{ext}"
            self.output_video_path = os.path.join(outputs_dir, output_filename)
            logger.info(f"[VIDEO_WRITER] Output video path: {self.output_video_path}")
            
            # Validate video properties before initialization
            if self.video_width <= 0 or self.video_height <= 0:
                logger.error(f"[VIDEO_WRITER] Invalid video dimensions: {self.video_width}x{self.video_height}")
                return False
            
            if self.video_fps <= 0:
                logger.warning(f"[VIDEO_WRITER] Invalid FPS: {self.video_fps}, using default 30.0")
                self.video_fps = 30.0
            
            # List of codecs to try in order of preference
            codec_options = [
                ('mp4v', 'MP4V'),  # Most compatible
                ('XVID', 'XVID'),  # Good fallback
                ('MJPG', 'MJPG'),  # Motion JPEG, widely supported
                ('X264', 'H264'),  # H.264 codec
                ('avc1', 'AVC1'),  # Another H.264 variant
            ]
            
            self.video_writer = None
            initialization_successful = False
            
            for codec_name, codec_desc in codec_options:
                try:
                    logger.info(f"[VIDEO_WRITER] Attempting initialization with {codec_desc} codec ({codec_name})")
                    
                    # Create fourcc code
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)  # pyright: ignore[reportAttributeAccessIssue]
                    
                    # Initialize video writer
                    self.video_writer = cv2.VideoWriter(
                        self.output_video_path, 
                        fourcc, 
                        self.video_fps, 
                        (self.video_width, self.video_height)
                    )  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                    
                    # Test if writer was successfully opened
                    if self.video_writer and self.video_writer.isOpened():
                        logger.info(f"[VIDEO_WRITER] Successfully initialized with {codec_desc} codec")
                        logger.info(f"[VIDEO_WRITER] Output parameters - FPS: {self.video_fps}, Size: {self.video_width}x{self.video_height}")
                        initialization_successful = True
                        break
                    else:
                        logger.warning(f"[VIDEO_WRITER] Failed to initialize with {codec_desc} codec")
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        
                except Exception as e:
                    logger.warning(f"[VIDEO_WRITER] Exception with {codec_desc} codec: {e}")
                    if self.video_writer:
                        try:
                            self.video_writer.release()
                        except:
                            pass
                        self.video_writer = None
                    continue
            
            if not initialization_successful:
                logger.error("[VIDEO_WRITER] Failed to initialize video writer with any codec")
                logger.warning("[VIDEO_WRITER] Processed video will not be saved")
                self.video_writer = None
                self.output_video_path = None
                return False
            
            # Validate writer by attempting to write a test frame
            try:
                test_frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
                self.video_writer.write(test_frame)
                logger.info("[VIDEO_WRITER] Test frame write successful")
            except Exception as e:
                logger.error(f"[VIDEO_WRITER] Test frame write failed: {e}")
                self._cleanup_video_writer()
                return False
            
            logger.info(f"[VIDEO_WRITER] Video writer initialization completed successfully")
            logger.info(f"[VIDEO_WRITER] Output will be saved to: {self.output_video_path}")
            return True
            
        except Exception as e:
            logger.error(f"[VIDEO_WRITER] Critical error during video writer initialization: {e}")
            traceback.print_exc()
            self._cleanup_video_writer()
            return False

    def _cleanup_video_writer(self):
        """
        Clean up video writer resources safely.
        """
        try:
            if self.video_writer:
                logger.info("[VIDEO_WRITER] Cleaning up video writer resources")
                try:
                    self.video_writer.release()
                    logger.info("[VIDEO_WRITER] Video writer released successfully")
                except Exception as e:
                    logger.warning(f"[VIDEO_WRITER] Error releasing video writer: {e}")
                finally:
                    self.video_writer = None
            
            # Check if output file was created and log its status
            if self.output_video_path and os.path.exists(self.output_video_path):
                file_size = os.path.getsize(self.output_video_path)
                if file_size > 0:
                    logger.info(f"[VIDEO_WRITER] Output video saved: {self.output_video_path} ({file_size} bytes)")
                else:
                    logger.warning(f"[VIDEO_WRITER] Output video file is empty: {self.output_video_path}")
                    try:
                        os.remove(self.output_video_path)
                        logger.info("[VIDEO_WRITER] Removed empty output file")
                    except Exception as e:
                        logger.warning(f"[VIDEO_WRITER] Failed to remove empty output file: {e}")
            elif self.output_video_path:
                logger.warning(f"[VIDEO_WRITER] Output video file was not created: {self.output_video_path}")
                
        except Exception as e:
            logger.error(f"[VIDEO_WRITER] Error during video writer cleanup: {e}")

    def _reinitialize_video_writer(self):
        """
        Reinitialize video writer if it becomes corrupted or fails.
        """
        try:
            logger.info("[VIDEO_WRITER] Attempting to reinitialize video writer")
            
            # Store current state
            current_output_path = self.output_video_path
            current_model_name = self.model_name
            current_video_source = self.video_source
            
            # Clean up current writer
            self._cleanup_video_writer()
            
            # Reinitialize with same parameters
            success = self._initialize_video_writer(current_video_source, current_model_name)
            
            if success:
                logger.info("[VIDEO_WRITER] Video writer reinitialization successful")
                return True
            else:
                logger.error("[VIDEO_WRITER] Video writer reinitialization failed")
                return False
                
        except Exception as e:
            logger.error(f"[VIDEO_WRITER] Error during video writer reinitialization: {e}")
            return False

    def _handle_video_writer_failure(self, frame_index, error):
        """
        Handle video writer failures with recovery attempts.
        
        Args:
            frame_index (int): Current frame index where failure occurred
            error (Exception): The error that occurred
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        try:
            logger.warning(f"[VIDEO_WRITER] Handling video writer failure at frame {frame_index}: {error}")
            
            # Increment error counter
            self.error_stats['video_write_errors'] += 1
            
            # If we've had too many write errors, try to reinitialize
            if self.error_stats['video_write_errors'] % 10 == 0:
                logger.info(f"[VIDEO_WRITER] Attempting recovery after {self.error_stats['video_write_errors']} write errors")
                
                # Try to reinitialize the video writer
                if self._reinitialize_video_writer():
                    logger.info("[VIDEO_WRITER] Video writer recovery successful")
                    return True
                else:
                    logger.error("[VIDEO_WRITER] Video writer recovery failed")
                    
            # If we've had too many errors overall, disable video writing
            if self.error_stats['video_write_errors'] > 50:
                logger.error(f"[VIDEO_WRITER] Too many video write errors ({self.error_stats['video_write_errors']}), disabling video output")
                self._cleanup_video_writer()
                return False
                
            return False
            
        except Exception as e:
            logger.error(f"[VIDEO_WRITER] Error handling video writer failure: {e}")
            return False

    def process_frames(self):
        logger.info("[PROCESS] Starting process_frames method")
        
        start_time = time.time()
        frame_count = 0
        last_frame_time = time.time()
        
        # Use the actual FPS from the video file or fallback
        target_time_per_frame = 1.0 / self.video_fps
        logger.info(f"[PROCESS] Processing at target FPS: {self.video_fps:.2f} (target time per frame: {target_time_per_frame:.4f}s)")
        logger.info(f"[PROCESS] Counter available: {self.counter is not None}, Stop event set: {self.stop_event.is_set()}")

        while not self.stop_event.is_set():
            try:
                # Log queue status before getting frame
                queue_size = self.frame_queue.qsize()
                logger.debug(f"[PROCESS] Queue size before get: {queue_size}/{self.frame_queue.maxsize}")
                
                # Get frame from queue with timeout
                frame_index, frame = self.frame_queue.get(timeout=2)  # Increased timeout for longer videos
                frame_count += 1
                self.frames_processed = frame_count

                # Log frame processing start
                logger.debug(f"[PROCESS] Starting to process frame {frame_index} (internal count: {frame_count})")
                
                # Control frame processing speed to match video FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed < target_time_per_frame:
                    sleep_time = target_time_per_frame - elapsed
                    if sleep_time > 0:
                        logger.debug(f"[PROCESS] Sleeping {sleep_time:.4f}s to match FPS timing")
                        time.sleep(sleep_time)
                
                last_frame_time = time.time()

                # Update heartbeat to detect stalled processing
                current_time = time.time()
                if current_time - self.last_heartbeat > self.heartbeat_interval:
                    self.last_heartbeat = current_time
                    logger.debug(f"[HEARTBEAT] Processing alive at frame {frame_count}/{self.total_frames} ({(frame_count/self.total_frames*100):.1f}%)")
                
                # Enhanced progress logging with detailed tracking
                if frame_count % self.logging_config['debug_log_interval'] == 0:
                    logger.debug(f"[PROCESS] Processing frame {frame_count} (index: {frame_index})")
                
                # Log comprehensive progress at regular intervals
                if frame_count % self.logging_config['progress_log_interval'] == 0:
                    self._log_detailed_progress()
                    
                    # Check if we're approaching completion
                    if frame_count >= self.total_frames - 5:
                        logger.info(f"[PROCESS] Approaching completion - Processed: {frame_count}, Total: {self.total_frames}")
                        # Update completion status to check all methods
                        if self._update_video_completion_status():
                            logger.info("[PROCESS] Video completion detected during progress check")
                            break

                # Process frame with counter
                if self.counter:
                    processing_success = False
                    processing_attempts = 0
                    max_processing_attempts = 3
                    
                    while not processing_success and processing_attempts < max_processing_attempts:
                        try:
                            processing_attempts += 1
                            process_start = time.time()
                            logger.debug(f"[PROCESS] Starting counter processing for frame {frame_index} (attempt {processing_attempts})")
                            
                            # Validate frame before processing
                            if frame is None or frame.size == 0:
                                raise ValueError(f"Invalid frame data: frame is {'None' if frame is None else 'empty'}")
                            
                            if len(frame.shape) != 3 or frame.shape[2] != 3:
                                raise ValueError(f"Invalid frame shape: {frame.shape}, expected 3-channel image")
                            
                            result = self.counter.process(frame)
                            process_time = time.time() - process_start
                            
                            # Validate processing result
                            if result is None:
                                raise ValueError("Counter returned None result")
                            
                            if not hasattr(result, 'plot_im') or result.plot_im is None:
                                logger.warning(f"[PROCESS] No annotated frame returned for frame {frame_index}, using original frame")
                                annotated_frame = frame.copy()
                            else:
                                annotated_frame = result.plot_im
                            
                            logger.debug(f"[PROCESS] Counter processing completed for frame {frame_index} in {process_time:.4f}s")
                            processing_success = True
                            
                            # Log processing performance
                            if frame_count % 25 == 0:  # Log performance every 25 frames
                                logger.info(f"[PERFORMANCE] Frame {frame_count} processing time: {process_time:.4f}s, "
                                          f"Objects detected: {getattr(result, 'total_tracks', 0)}, In count: {getattr(result, 'in_count', 0)}")
                            
                            # Write annotated frame to output video if writer is available
                            if self.video_writer and annotated_frame is not None:
                                write_success = False
                                write_attempts = 0
                                max_write_attempts = 2
                                
                                while not write_success and write_attempts < max_write_attempts:
                                    try:
                                        write_attempts += 1
                                        write_start = time.time()
                                        
                                        # Validate annotated frame
                                        if annotated_frame is None or annotated_frame.size == 0:
                                            logger.warning(f"[VIDEO_WRITE] Invalid annotated frame for frame {frame_index}, using original frame")
                                            annotated_frame = frame.copy()
                                        
                                        # Ensure frame has correct dimensions
                                        if annotated_frame.shape[:2][::-1] != (self.video_width, self.video_height):
                                            logger.debug(f"[VIDEO_WRITE] Resizing frame {frame_index} from {annotated_frame.shape[:2][::-1]} to ({self.video_width}, {self.video_height})")
                                            annotated_frame = cv2.resize(annotated_frame, (self.video_width, self.video_height))  # pyright: ignore[reportAttributeAccessIssue]
                                        
                                        self.video_writer.write(annotated_frame)
                                        write_time = time.time() - write_start
                                        write_success = True
                                        
                                        if frame_count % 50 == 0:  # Log write performance less frequently
                                            logger.debug(f"[VIDEO_WRITE] Frame {frame_count} write time: {write_time:.4f}s")
                                            
                                    except Exception as e:
                                        logger.error(f"[VIDEO_WRITE] Error writing frame {frame_index} to output video "
                                                   f"(attempt {write_attempts}/{max_write_attempts}): {e}")
                                        logger.debug(f"[VIDEO_WRITE] Write error context - Frame shape: {annotated_frame.shape if annotated_frame is not None else 'None'}, "
                                                   f"Writer available: {self.video_writer is not None}")
                                        
                                        # Handle video writer failure with recovery attempts
                                        recovery_successful = self._handle_video_writer_failure(frame_index, e)
                                        
                                        if write_attempts >= max_write_attempts:
                                            if recovery_successful:
                                                logger.info(f"[VIDEO_WRITE] Video writer recovered, retrying frame {frame_index}")
                                                write_attempts = 0  # Reset attempts after successful recovery
                                                continue
                                            else:
                                                logger.error(f"[VIDEO_WRITE] Failed to write frame {frame_index} after {max_write_attempts} attempts, skipping")
                                                break
                                        else:
                                            time.sleep(0.01)  # Brief pause before retry
                            
                            # Log detection results
                            if frame_count % 10 == 0:  # Log every 10 frames for longer videos
                                in_count = getattr(result, 'in_count', 0)
                                out_count = getattr(result, 'out_count', 0)
                                total_tracks = getattr(result, 'total_tracks', 0)
                                logger.info(f"[DETECTION] Frame {frame_count}: in_count={in_count}, "
                                          f"out_count={out_count}, "
                                          f"total_tracks={total_tracks}")
                                
                                # Log classwise counts if available
                                if hasattr(result, 'classwise_count') and result.classwise_count:
                                    try:
                                        class_counts = []
                                        for class_name, count in result.classwise_count.items():
                                            if isinstance(count, dict):
                                                # Handle nested count structure
                                                total = sum(count.values()) if isinstance(count, dict) else count
                                                class_counts.append(f"{class_name}:{total}")
                                            else:
                                                class_counts.append(f"{class_name}:{count}")
                                        logger.info(f"[DETECTION] Classwise counts: {', '.join(class_counts)}")
                                    except Exception as e:
                                        logger.warning(f"[DETECTION] Error logging classwise counts: {e}")
                            
                            # Store the last valid result
                            self.last_valid_result = self.latest_result
                            
                            # Enhanced result logging with error handling
                            try:
                                current_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                                self.latest_result = {
                                    'frame': annotated_frame,
                                    'in_count': getattr(result, 'in_count', 0),
                                    'total_tracks': getattr(result, 'total_tracks', 0),
                                    'classwise_counts': getattr(result, 'classwise_count', {}),
                                    'fps': current_fps,
                                    'frame_count': frame_count,
                                    'timestamp': time.time(),
                                    'model_name': self.model_name,
                                    'model_metadata': {
                                        'model_path': getattr(self.counter, 'model_path', None) if self.counter else None,
                                        'fallback_used': getattr(self.counter, 'fallback_used', False) if self.counter else False,
                                        'model_file_name': self.model_name
                                    }
                                }
                            except Exception as e:
                                logger.warning(f"[RESULT] Error creating result object for frame {frame_count}: {e}")
                                # Create minimal result to prevent pipeline failure
                                self.latest_result = {
                                    'frame': annotated_frame if 'annotated_frame' in locals() else frame,
                                    'in_count': 0,
                                    'total_tracks': 0,
                                    'classwise_counts': {},
                                    'fps': 0,
                                    'frame_count': frame_count,
                                    'timestamp': time.time(),
                                    'model_name': self.model_name,
                                    'model_metadata': {
                                        'model_path': getattr(self.counter, 'model_path', None) if self.counter else None,
                                        'fallback_used': getattr(self.counter, 'fallback_used', False) if self.counter else False,
                                        'model_file_name': self.model_name
                                    }
                                }
                            
                            # Log successful frame processing
                            if frame_count % 25 == 0:
                                in_count = getattr(result, 'in_count', 0)
                                total_tracks = getattr(result, 'total_tracks', 0)
                                logger.info(f"[FRAME] Processed frame {frame_count}, in_count: {in_count}, tracks: {total_tracks}")
                                if hasattr(annotated_frame, 'shape'):
                                    logger.debug(f"[FRAME] Annotated frame shape: {annotated_frame.shape}")
                                else:
                                    logger.warning(f"[FRAME] Annotated frame is not a numpy array: {type(annotated_frame)}")
                                    
                        except Exception as e:
                            self.error_stats['frame_processing_errors'] += 1
                            
                            # Check if this is a CUDA error
                            is_cuda_error = 'CUDA' in str(e) or 'no kernel image' in str(e)
                            
                            if is_cuda_error:
                                self.error_stats['cuda_errors'] += 1
                                self.error_stats['recent_cuda_errors'] += 1
                                logger.warning(f"[CUDA_ERROR] CUDA error processing frame {frame_count} "
                                             f"(attempt {processing_attempts}/{max_processing_attempts}): {e}")
                                logger.info("[CUDA_ERROR] This is a GPU compatibility issue, not an end-of-video condition")
                                logger.info(f"[CUDA_ERROR] Total CUDA errors: {self.error_stats['cuda_errors']}")
                            else:
                                logger.error(f"[ERROR] Error processing frame {frame_count} with counter "
                                           f"(attempt {processing_attempts}/{max_processing_attempts}): {e}")
                            
                            logger.debug(f"[ERROR] Processing error context - Frame shape: {frame.shape if frame is not None else 'None'}, "
                                       f"Counter available: {self.counter is not None}, "
                                       f"Total processing errors: {self.error_stats['frame_processing_errors']}, "
                                       f"CUDA error: {is_cuda_error}")
                            
                            if processing_attempts == 1:  # Only print full traceback on first attempt
                                traceback.print_exc()
                            
                            if processing_attempts >= max_processing_attempts:
                                if is_cuda_error:
                                    logger.warning(f"[CUDA_ERROR] Failed to process frame {frame_count} after {max_processing_attempts} attempts due to CUDA issues, creating CPU fallback result")
                                else:
                                    logger.error(f"[ERROR] Failed to process frame {frame_count} after {max_processing_attempts} attempts, creating fallback result")
                                
                                # Create fallback result to prevent pipeline failure
                                try:
                                    self.latest_result = {
                                        'frame': frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                                        'in_count': self.latest_result.get('in_count', 0) if self.latest_result else 0,
                                        'total_tracks': 0,
                                        'classwise_counts': {},
                                        'fps': frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
                                        'frame_count': frame_count,
                                        'timestamp': time.time(),
                                        'model_name': self.model_name,
                                        'model_metadata': {
                                            'model_path': getattr(self.counter, 'model_path', None) if self.counter else None,
                                            'fallback_used': getattr(self.counter, 'fallback_used', False) if self.counter else False,
                                            'model_file_name': self.model_name
                                        },
                                        'cuda_error': is_cuda_error  # Flag to indicate CUDA issues
                                    }
                                    processing_success = True  # Mark as success to continue processing
                                except Exception as fallback_error:
                                    logger.error(f"[ERROR] Failed to create fallback result: {fallback_error}")
                                    continue  # Skip this frame entirely
                            else:
                                # Brief pause before retry
                                time.sleep(0.05)
                else:
                    logger.warning("[WARNING] Counter not initialized")
                    # Create minimal result when counter is not available
                    self.latest_result = {
                        'frame': frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                        'in_count': 0,
                        'total_tracks': 0,
                        'classwise_counts': {},
                        'fps': frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
                        'frame_count': frame_count,
                        'timestamp': time.time(),
                        'model_name': self.model_name,
                        'model_metadata': {
                            'model_path': None,
                            'fallback_used': False,
                            'model_file_name': self.model_name
                        }
                    }
                
            except queue.Empty:
                # Check if we've reached the end of video using comprehensive detection
                if self._update_video_completion_status():
                    logger.info("[END] End of video reached via comprehensive detection, stopping processing")
                    break
                # No frames available, continue waiting
                logger.debug("[PROCESS] No frames available in queue, waiting...")
                continue
            except Exception as e:
                logger.error(f"[ERROR] Error processing frame {frame_count}: {e}")
                traceback.print_exc()
        
        # Log final processing statistics
        total_elapsed = time.time() - start_time
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"[FINAL] Processing thread finished. Total frames processed: {frame_count}, "
                  f"Total time: {total_elapsed:.2f}s, Average FPS: {avg_fps:.2f}")
        logger.info(f"[FINAL] Total frames reported by capture: {self.frames_processed}")
        
        # Mark processing thread as finished and update completion status
        self.video_completion_flags['processing_thread_finished'] = True
        self._update_video_completion_status()
        
        logger.info(f"[FINAL] Final completion status: {self.get_completion_status()}")
    
    def start_processing(self, video_source, model_name=None):
        """
        Start video processing
        
        Args:
            video_source (str): Path to video file
            model_name (str, optional): Name of model file to use
        """
        logger.info(f"[START] Starting video processing for: {video_source} with model: {model_name}")
        
        if self.processing:
            logger.warning("[START] Video processing is already in progress.")
            return False, "Processing already in progress"

        logger.info(f"[START] Opening video source: {video_source}")
        self.cap = cv2.VideoCapture(video_source)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.video_source = video_source  # Store for potential reinitialization
        self.model_name = model_name  # Store model name for database saving
        self.start_time = time.time()  # Track start time
        
        if not self.cap.isOpened():
            logger.error(f"[START] Failed to open video source: {video_source}")
            return False, "Failed to open video source"
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pyright: ignore[reportAttributeAccessIssue]
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)  # pyright: ignore[reportAttributeAccessIssue]
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # pyright: ignore[reportAttributeAccessIssue]
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # pyright: ignore[reportAttributeAccessIssue]
        
        logger.info(f"[START] Video properties - Frames: {self.total_frames}, FPS: {self.video_fps}, Size: {self.video_width}x{self.video_height}")

        # Validate FPS
        if self.video_fps <= 0:
            self.video_fps = 30.0  # Default fallback
            logger.warning(f"[START] Invalid FPS detected, using default: {self.video_fps}")
        
        # Set up video writer for output with enhanced error handling
        self._initialize_video_writer(video_source, model_name)
        
        # Initialize progress tracking
        self.progress_tracking['start_time'] = time.time()
        self._log_processing_stage_transition('initialization', 'start')

        # Set buffer size
        logger.info("[START] Setting video capture buffer size to 1")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # pyright: ignore[reportAttributeAccessIssue]

        self.processing = True
        self.stop_event.clear() 
        self.end_of_video = False
        self.processing_complete = False
        self.frames_processed = 0
        self.consecutive_read_failures = 0  # Reset failure counter
        
        # Initialize progress tracking
        self.progress_tracking['start_time'] = time.time()
        self.progress_tracking['last_progress_update'] = 0
        self.progress_tracking['frames_per_second_history'] = []
        self.progress_tracking['estimated_time_remaining'] = 0
        
        # Reset processing stages
        for stage in self.progress_tracking['processing_stages']:
            self.progress_tracking['processing_stages'][stage] = {'start': None, 'end': None, 'duration': 0}
        
        # Reset performance metrics
        self.progress_tracking['performance_metrics'] = {
            'avg_frame_processing_time': 0,
            'avg_capture_fps': 0,
            'avg_processing_fps': 0,
            'peak_processing_fps': 0,
            'memory_usage_mb': 0
        }
        
        # Log initialization stage
        self._log_processing_stage_transition('initialization', 'start')
        
        # Reset completion flags and status
        self.video_completion_flags = {
            'frame_position_reached': False,
            'read_failures_exceeded': False,
            'total_frames_processed': False,
            'capture_thread_finished': False,
            'processing_thread_finished': False
        }
        self.video_completion_status = {
            'completion_method': None,
            'completion_timestamp': None,
            'final_frame_position': 0,
            'final_frames_processed': 0,
            'completion_reason': None
        }
        
        # Initialize counter with model name and enhanced error handling
        counter_init_attempts = 0
        max_counter_init_attempts = 3
        counter_initialized = False
        
        while not counter_initialized and counter_init_attempts < max_counter_init_attempts:
            try:
                counter_init_attempts += 1
                logger.info(f"[START] Initializing counter with model: {model_name} (attempt {counter_init_attempts})")
                from .counter import object_count
                self.counter = object_count(model_name=model_name)
                
                # Validate counter initialization
                if self.counter is None:
                    raise ValueError("Counter initialization returned None")
                
                # Update region points to be within video bounds if possible
                # This is a workaround since we can't pass video dimensions to object_count
                if hasattr(self.counter, 'region') and self.video_width > 0 and self.video_height > 0:
                    try:
                        # Set region to be a vertical line near the right edge of the frame
                        region_x = max(int(self.video_width * 0.8), self.video_width - 200)
                        new_region = [(region_x, 0), (region_x + 9, self.video_height)]
                        self.counter.region = new_region
                        logger.info(f"[START] Updated counter region to: {new_region} for video size {self.video_width}x{self.video_height}")
                    except Exception as e:
                        logger.warning(f"[START] Could not update counter region: {e}")
                
                logger.info(f"[START] Counter initialized successfully with region: {getattr(self.counter, 'region', 'N/A')}")
                counter_initialized = True
                
            except Exception as e:
                logger.error(f"[START] Failed to initialize counter (attempt {counter_init_attempts}): {e}")
                if counter_init_attempts == 1:  # Only print full traceback on first attempt
                    traceback.print_exc()
                
                if counter_init_attempts >= max_counter_init_attempts:
                    logger.error(f"[START] Failed to initialize counter after {max_counter_init_attempts} attempts")
                    self._cleanup()
                    return False, f"Failed to initialize counter after {max_counter_init_attempts} attempts: {e}"
                else:
                    logger.info(f"[START] Retrying counter initialization in 1 second...")
                    time.sleep(1.0)

        # Start capture and processing threads with error handling
        try:
            logger.info("[START] Starting capture and processing threads")
            
            # Create thread wrapper functions with error handling
            def capture_thread_wrapper():
                try:
                    logger.info("[CAPTURE_THREAD] Starting capture thread execution")
                    self.capture_frames(video_source)
                    logger.info("[CAPTURE_THREAD] Capture thread completed normally")
                except KeyboardInterrupt:
                    logger.info("[CAPTURE_THREAD] Capture thread interrupted by user")
                    self.stop_event.set()
                except SystemExit:
                    logger.info("[CAPTURE_THREAD] Capture thread received system exit")
                    self.stop_event.set()
                except Exception as e:
                    logger.error(f"[CAPTURE_THREAD] Capture thread failed with exception: {e}")
                    logger.error(f"[CAPTURE_THREAD] Exception type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    self.stop_event.set()
                finally:
                    logger.info("[CAPTURE_THREAD] Capture thread wrapper finished")
            
            def process_thread_wrapper():
                try:
                    logger.info("[PROCESS_THREAD] Starting process thread execution")
                    self.process_frames()
                    logger.info("[PROCESS_THREAD] Process thread completed normally")
                except KeyboardInterrupt:
                    logger.info("[PROCESS_THREAD] Process thread interrupted by user")
                    self.stop_event.set()
                except SystemExit:
                    logger.info("[PROCESS_THREAD] Process thread received system exit")
                    self.stop_event.set()
                except Exception as e:
                    logger.error(f"[PROCESS_THREAD] Process thread failed with exception: {e}")
                    logger.error(f"[PROCESS_THREAD] Exception type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    self.stop_event.set()
                finally:
                    logger.info("[PROCESS_THREAD] Process thread wrapper finished")
            
            self.capture_thread = threading.Thread(target=capture_thread_wrapper, name="CaptureThread")
            self.process_thread = threading.Thread(target=process_thread_wrapper, name="ProcessThread")
            
            # Start threads with error handling
            try:
                self.capture_thread.start()
                logger.info("[START] Capture thread started successfully")
            except Exception as e:
                logger.error(f"[START] Failed to start capture thread: {e}")
                self._cleanup()
                return False, f"Failed to start capture thread: {e}"
            
            try:
                self.process_thread.start()
                logger.info("[START] Process thread started successfully")
            except Exception as e:
                logger.error(f"[START] Failed to start process thread: {e}")
                # Stop capture thread if process thread failed
                self.stop_event.set()
                if self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=5.0)
                self._cleanup()
                return False, f"Failed to start process thread: {e}"
            
            # Give threads a moment to initialize before checking
            time.sleep(0.5)  # Increased pause to let threads initialize properly
            
            # Check if threads are still running after initialization
            capture_alive = self.capture_thread.is_alive()
            process_alive = self.process_thread.is_alive()
            
            logger.info(f"[START] Thread status after initialization - Capture: {capture_alive}, Process: {process_alive}")
            
            if not capture_alive:
                logger.error("[START] Capture thread died during initialization")
                self._cleanup()
                return False, "Capture thread failed during initialization"
            
            if not process_alive:
                logger.error("[START] Process thread died during initialization")
                self._cleanup()
                return False, "Process thread failed during initialization"
            
            logger.info("[START] Processing threads started and verified successfully")
            
            # Log initialization completion and start capture stage
            self._log_processing_stage_transition('initialization', 'end')
            self._log_processing_stage_transition('capture', 'start')
            self._log_processing_stage_transition('processing', 'start')
            
            # Log initial progress
            self._log_detailed_progress(force_log=True)
            
            return True, "Processing started"
            
        except Exception as e:
            logger.error(f"[START] Critical error starting processing threads: {e}")
            traceback.print_exc()
            self._cleanup()
            return False, f"Critical error starting threads: {e}"

    def stop(self, save_partial_results=True):
        """
        Stop video processing with enhanced partial result saving
        
        Args:
            save_partial_results (bool): Whether to save partial results when stopping
        """
        logger.info(f"[STOP] Stopping video processing. Current state - processing: {self.processing}, "
                   f"frames_processed: {self.frames_processed}, total_frames: {self.total_frames}")
        
        if not self.processing:
            logger.info("[STOP] No processing to stop")
            return {
                'success': True,
                'message': 'No processing to stop',
                'partial_results': None,
                'processing_stats': None
            }
        
        # Mark as manually stopped for tracking
        self.manually_stopped = True
        self.stop_requested_time = time.time()
        
        logger.info("[STOP] Setting stop event")
        self.stop_event.set()
        
        # Capture current processing state before stopping threads
        partial_results = None
        processing_stats = None
        
        if save_partial_results:
            logger.info("[STOP] Capturing partial results before stopping threads")
            partial_results = self._capture_partial_results()
            processing_stats = self._capture_processing_stats()

        # Wait for threads to finish with longer timeout
        threads_stopped_successfully = True
        
        if self.capture_thread and self.capture_thread.is_alive():
            logger.info("[STOP] Waiting for capture thread to finish...")
            self.capture_thread.join(timeout=15.0)  # Increased timeout for longer videos
            if self.capture_thread.is_alive():
                logger.warning("[STOP] Capture thread did not stop gracefully")
                threads_stopped_successfully = False
            else:
                logger.info("[STOP] Capture thread stopped successfully")
        
        if self.process_thread and self.process_thread.is_alive():
            logger.info("[STOP] Waiting for process thread to finish...")
            self.process_thread.join(timeout=15.0)  # Increased timeout for longer videos
            if self.process_thread.is_alive():
                logger.warning("[STOP] Process thread did not stop gracefully")
                threads_stopped_successfully = False
            else:
                logger.info("[STOP] Process thread stopped successfully")
        
        # Log final progress and statistics
        self._log_detailed_progress(force_log=True)
        
        # Log stage transitions
        self._log_processing_stage_transition('capture', 'end')
        self._log_processing_stage_transition('processing', 'end')
        self._log_processing_stage_transition('cleanup', 'start')
        
        # Log comprehensive final statistics
        if self.progress_tracking['start_time']:
            total_time = time.time() - self.progress_tracking['start_time']
            logger.info(f"[STOP] Total processing time: {total_time:.2f}s")
            
            if self.total_frames > 0:
                progress_percent = (self.frames_processed/self.total_frames)*100
                avg_fps = self.frames_processed / total_time if total_time > 0 else 0
                logger.info(f"[STOP] Final progress: {progress_percent:.1f}% "
                          f"({self.frames_processed}/{self.total_frames} frames)")
                logger.info(f"[STOP] Average processing FPS: {avg_fps:.2f}")
                logger.info(f"[STOP] Peak processing FPS: {self.progress_tracking['performance_metrics']['peak_processing_fps']:.2f}")
                
                # Log stage durations
                for stage_name, stage_info in self.progress_tracking['processing_stages'].items():
                    if stage_info['duration'] > 0:
                        logger.info(f"[STOP] {stage_name.capitalize()} stage duration: {stage_info['duration']:.2f}s")
        
        self._cleanup()
        
        # Return comprehensive stop result
        stop_result = {
            'success': threads_stopped_successfully,
            'message': 'Processing stopped successfully' if threads_stopped_successfully else 'Processing stopped with warnings',
            'partial_results': partial_results,
            'processing_stats': processing_stats,
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'completion_percentage': (self.frames_processed / self.total_frames * 100) if self.total_frames > 0 else 0,
            'manually_stopped': True
        }
        
        logger.info(f"[STOP] Stop operation completed. Processed {self.frames_processed}/{self.total_frames} frames "
                   f"({stop_result['completion_percentage']:.1f}%)")
        
        return stop_result

    def _cleanup(self):
        """Clean up resources"""
        logger.info("[CLEANUP] Starting cleanup process")
        
        try:
            if self.cap:
                logger.info("[CLEANUP] Releasing video capture")
                self.cap.release()
                self.cap = None
            
            # Release video writer if it exists
            self._cleanup_video_writer()
            
            self.processing = False
            self.stop_event.clear()
            self.consecutive_read_failures = 0
            
            # Clear frame queue
            queue_size = self.frame_queue.qsize()
            if queue_size > 0:
                logger.info(f"[CLEANUP] Clearing frame queue with {queue_size} remaining items")
                cleared_count = 0
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                        cleared_count += 1
                    except:
                        break
                logger.info(f"[CLEANUP] Cleared {cleared_count} items from frame queue")
            
            logger.info("[CLEANUP] Cleanup completed successfully")
            
            # Log cleanup stage completion
            self._log_processing_stage_transition('cleanup', 'end')
            
        except Exception as e:
            logger.error(f"[CLEANUP] Error during cleanup: {e}")
            traceback.print_exc()
    
    def get_latest_result(self):
        """Get the latest processing result"""
        result = self.latest_result or self.last_valid_result
        if result:
            # Add comprehensive progress information
            progress_info = self._calculate_accurate_progress()
            result.update({
                'progress': progress_info['percentage'],
                'progress_info': progress_info,
                'is_complete': self.processing_complete,
                'completion_status': self.get_completion_status()
            })
            
            # Add debug info at reduced frequency
            if self.frames_processed % self.logging_config['performance_log_interval'] == 0:
                logger.debug(f"[STATUS] Result: frames_processed={self.frames_processed}/{self.total_frames}, "
                           f"progress={progress_info['percentage']:.1f}%, "
                           f"fps={progress_info['current_fps']:.1f}, "
                           f"complete={self.processing_complete}")
        return result
    
    def _detect_video_completion(self):
        """
        Comprehensive end-of-video detection using multiple methods
        Returns tuple: (is_complete, completion_method, completion_reason)
        """
        completion_methods = []
        
        # Method 1: Frame position detection (more conservative)
        if self.cap and self.cap.isOpened():
            current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # pyright: ignore[reportAttributeAccessIssue]
            # Only consider complete if we're very close to the end AND have processed significant frames
            if (self.total_frames > 0 and 
                current_frame_pos >= self.total_frames - 1 and 
                self.frames_processed >= (self.total_frames * 0.9)):  # Must have processed at least 90%
                completion_methods.append(('frame_position', f'Frame position {current_frame_pos} >= total frames {self.total_frames} (processed {self.frames_processed} frames)'))
                self.video_completion_flags['frame_position_reached'] = True
        
        # Method 2: Consecutive read failures (only if we haven't processed significant frames)
        # Don't trigger completion on read failures if we've successfully processed frames
        # This prevents CUDA processing errors from being mistaken for end-of-video
        if (self.consecutive_read_failures >= self.max_consecutive_read_failures and 
            self.frames_processed < (self.total_frames * 0.1)):  # Only if less than 10% processed
            completion_methods.append(('read_failures', f'Consecutive read failures {self.consecutive_read_failures} >= max {self.max_consecutive_read_failures}'))
            self.video_completion_flags['read_failures_exceeded'] = True
        
        # Method 3: Total frames processed
        if self.total_frames > 0 and self.frames_processed >= self.total_frames - 1:
            completion_methods.append(('total_frames', f'Frames processed {self.frames_processed} >= total frames {self.total_frames}'))
            self.video_completion_flags['total_frames_processed'] = True
        
        # Method 4: Capture thread finished (only check after some processing has occurred)
        if (self.capture_thread and not self.capture_thread.is_alive() and 
            (self.frames_processed > 0 or time.time() - (self.start_time or 0) > 10)):
            completion_methods.append(('capture_thread', 'Capture thread has finished'))
            self.video_completion_flags['capture_thread_finished'] = True
        
        # Method 5: Processing thread finished (only check after some processing has occurred)
        if (self.process_thread and not self.process_thread.is_alive() and 
            (self.frames_processed > 0 or time.time() - (self.start_time or 0) > 10)):
            completion_methods.append(('processing_thread', 'Processing thread has finished'))
            self.video_completion_flags['processing_thread_finished'] = True
        
        # Don't mark as complete if we have recent CUDA errors (they might resolve)
        # This prevents CUDA compatibility issues from being mistaken for end-of-video
        if hasattr(self, 'error_stats') and self.error_stats.get('recent_cuda_errors', 0) > 10:
            logger.info(f"[COMPLETION] Delaying completion detection due to {self.error_stats['recent_cuda_errors']} recent CUDA errors")
            # Reset recent CUDA errors counter to give it another chance
            self.error_stats['recent_cuda_errors'] = 0
            return False, None, "CUDA errors preventing completion detection"
        
        # Determine if video is complete
        is_complete = len(completion_methods) > 0
        
        if is_complete and not self.video_completion_status['completion_method']:
            # Record the first completion detection
            primary_method = completion_methods[0]
            self.video_completion_status.update({
                'completion_method': primary_method[0],
                'completion_timestamp': time.time(),
                'final_frame_position': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap and self.cap.isOpened() else 0,  # pyright: ignore[reportAttributeAccessIssue]
                'final_frames_processed': self.frames_processed,
                'completion_reason': primary_method[1]
            })
            
            logger.info(f"[VIDEO_COMPLETION] Video completion detected via {primary_method[0]}: {primary_method[1]}")
            logger.info(f"[VIDEO_COMPLETION] All completion methods detected: {[method[0] for method in completion_methods]}")
            logger.info(f"[VIDEO_COMPLETION] Final status - Position: {self.video_completion_status['final_frame_position']}, "
                       f"Processed: {self.video_completion_status['final_frames_processed']}, "
                       f"Total: {self.total_frames}")
        
        return is_complete, completion_methods[0][0] if completion_methods else None, completion_methods[0][1] if completion_methods else None

    def _update_video_completion_status(self):
        """Update video completion flags and status"""
        is_complete, method, reason = self._detect_video_completion()
        
        if is_complete and not self.end_of_video:
            self.end_of_video = True
            self.processing_complete = True
            logger.info(f"[VIDEO_STATUS] Video marked as complete via {method}: {reason}")
            
            # Log all completion flags for debugging
            active_flags = [flag for flag, status in self.video_completion_flags.items() if status]
            logger.info(f"[VIDEO_STATUS] Active completion flags: {active_flags}")
        
        return is_complete

    def is_processing_complete(self):
        """Check if video processing is complete using comprehensive detection"""
        logger.debug(f"[COMPLETION] Checking completion status - "
                   f"end_of_video: {self.end_of_video}, "
                   f"frames_processed: {self.frames_processed}, "
                   f"total_frames: {self.total_frames}, "
                   f"processing_complete: {self.processing_complete}")
        
        # Update completion status using comprehensive detection
        self._update_video_completion_status()
        
        return self.processing_complete

    def get_completion_status(self):
        """Get detailed completion status information"""
        return {
            'is_complete': self.processing_complete,
            'end_of_video': self.end_of_video,
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'completion_flags': self.video_completion_flags.copy(),
            'completion_status': self.video_completion_status.copy(),
            'progress_percentage': (self.frames_processed / self.total_frames * 100) if self.total_frames > 0 else 0,
            'error_stats': self.error_stats.copy()
        }

    def _check_processing_health(self):
        """Monitor processing health and detect stalled processing"""
        current_time = time.time()
        
        # Check if processing has stalled
        time_since_last_frame = current_time - self.last_successful_frame_time
        if time_since_last_frame > self.processing_health_check_interval:
            logger.warning(f"[HEALTH] Processing may be stalled - {time_since_last_frame:.1f}s since last successful frame")
            
            # Log current state for debugging
            logger.info(f"[HEALTH] Current state - Frames processed: {self.frames_processed}/{self.total_frames}, "
                       f"Consecutive failures: {self.consecutive_read_failures}, "
                       f"Queue size: {self.frame_queue.qsize()}")
            
            # Check thread health
            if self.capture_thread and not self.capture_thread.is_alive():
                logger.warning("[HEALTH] Capture thread has died unexpectedly")
            
            if self.process_thread and not self.process_thread.is_alive():
                logger.warning("[HEALTH] Process thread has died unexpectedly")
            
            return False
        
        return True

    def _handle_critical_error(self, error_type, error_message, exception=None):
        """Handle critical errors that may require processing termination"""
        logger.error(f"[CRITICAL] {error_type}: {error_message}")
        if exception:
            logger.error(f"[CRITICAL] Exception details: {exception}")
            traceback.print_exc()
        
        # Update error statistics
        if error_type == "frame_read":
            self.error_stats['frame_read_errors'] += 1
        elif error_type == "frame_processing":
            self.error_stats['frame_processing_errors'] += 1
        elif error_type == "video_write":
            self.error_stats['video_write_errors'] += 1
        
        # Check if we should attempt recovery or terminate
        total_errors = sum([
            self.error_stats['frame_read_errors'],
            self.error_stats['frame_processing_errors'],
            self.error_stats['video_write_errors']
        ])
        
        if total_errors > 1000:  # Threshold for critical error count
            logger.error(f"[CRITICAL] Too many errors ({total_errors}), processing may need to be terminated")
            return False
        
        return True

    def _update_processing_health(self):
        """Update processing health indicators"""
        self.last_successful_frame_time = time.time()
        
        # Reset consecutive failures on successful processing
        if self.consecutive_read_failures > 0:
            logger.debug(f"[HEALTH] Resetting consecutive failures from {self.consecutive_read_failures} to 0")
            self.consecutive_read_failures = 0

    def _calculate_accurate_progress(self):
        """Calculate accurate progress based on frames processed vs total frames"""
        if self.total_frames <= 0:
            return {
                'percentage': 0.0,
                'frames_processed': self.frames_processed,
                'total_frames': 0,
                'frames_remaining': 0,
                'estimated_time_remaining': 0,
                'current_fps': 0.0,
                'avg_fps': 0.0,
                'stage': 'unknown'
            }
        
        # Calculate basic progress
        progress_percentage = min((self.frames_processed / self.total_frames) * 100, 100.0)
        frames_remaining = max(0, self.total_frames - self.frames_processed)
        
        # Calculate FPS and time estimates
        current_time = time.time()
        elapsed_time = current_time - (self.progress_tracking['start_time'] or current_time)
        
        current_fps = 0.0
        avg_fps = 0.0
        estimated_time_remaining = 0
        
        if elapsed_time > 0:
            avg_fps = self.frames_processed / elapsed_time
            
            # Calculate current FPS from recent history
            if len(self.progress_tracking['frames_per_second_history']) > 0:
                recent_fps = self.progress_tracking['frames_per_second_history'][-5:]  # Last 5 measurements
                current_fps = sum(recent_fps) / len(recent_fps)
            else:
                current_fps = avg_fps
            
            # Estimate time remaining
            if current_fps > 0:
                estimated_time_remaining = frames_remaining / current_fps
        
        # Determine current processing stage
        stage = 'initialization'
        if self.processing:
            if self.frames_processed > 0:
                if progress_percentage >= 95:
                    stage = 'finalizing'
                else:
                    stage = 'processing'
            else:
                stage = 'starting'
        elif self.processing_complete:
            stage = 'complete'
        
        return {
            'percentage': round(progress_percentage, 2),
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'frames_remaining': frames_remaining,
            'estimated_time_remaining': round(estimated_time_remaining, 1),
            'current_fps': round(current_fps, 2),
            'avg_fps': round(avg_fps, 2),
            'stage': stage,
            'elapsed_time': round(elapsed_time, 1)
        }

    def _log_detailed_progress(self, force_log=False):
        """Log detailed progress information for user feedback"""
        current_time = time.time()
        
        # Check if we should log progress
        if not force_log:
            time_since_last_log = current_time - self.progress_tracking['last_progress_update']
            if time_since_last_log < self.progress_tracking['progress_update_interval']:
                return
        
        # Update last progress log time
        self.progress_tracking['last_progress_update'] = current_time
        
        # Calculate comprehensive progress
        progress_info = self._calculate_accurate_progress()
        
        # Update FPS history for better current FPS calculation
        if progress_info['current_fps'] > 0:
            self.progress_tracking['frames_per_second_history'].append(progress_info['current_fps'])
            # Keep only last 20 measurements
            if len(self.progress_tracking['frames_per_second_history']) > 20:
                self.progress_tracking['frames_per_second_history'] = self.progress_tracking['frames_per_second_history'][-20:]
        
        # Update performance metrics
        self.progress_tracking['performance_metrics']['avg_processing_fps'] = progress_info['avg_fps']
        if progress_info['current_fps'] > self.progress_tracking['performance_metrics']['peak_processing_fps']:
            self.progress_tracking['performance_metrics']['peak_processing_fps'] = progress_info['current_fps']
        
        # Format time remaining
        time_remaining_str = "unknown"
        if progress_info['estimated_time_remaining'] > 0:
            if progress_info['estimated_time_remaining'] < 60:
                time_remaining_str = f"{progress_info['estimated_time_remaining']:.1f}s"
            elif progress_info['estimated_time_remaining'] < 3600:
                minutes = int(progress_info['estimated_time_remaining'] // 60)
                seconds = int(progress_info['estimated_time_remaining'] % 60)
                time_remaining_str = f"{minutes}m {seconds}s"
            else:
                hours = int(progress_info['estimated_time_remaining'] // 3600)
                minutes = int((progress_info['estimated_time_remaining'] % 3600) // 60)
                time_remaining_str = f"{hours}h {minutes}m"
        
        # Log comprehensive progress information
        logger.info(f"[PROGRESS] {progress_info['percentage']:.1f}% complete "
                   f"({progress_info['frames_processed']}/{progress_info['total_frames']} frames)")
        logger.info(f"[PROGRESS] Stage: {progress_info['stage']}, "
                   f"Current FPS: {progress_info['current_fps']:.1f}, "
                   f"Avg FPS: {progress_info['avg_fps']:.1f}")
        logger.info(f"[PROGRESS] Elapsed: {progress_info['elapsed_time']:.1f}s, "
                   f"Remaining: {time_remaining_str}, "
                   f"Frames left: {progress_info['frames_remaining']}")
        
        # Log performance metrics periodically
        if self.frames_processed % self.logging_config['performance_log_interval'] == 0:
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        """Log detailed performance metrics for debugging"""
        try:
            import psutil
            import os
            
            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.progress_tracking['performance_metrics']['memory_usage_mb'] = round(memory_mb, 1)
            
            logger.info(f"[PERFORMANCE] Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            logger.debug("[PERFORMANCE] psutil not available for memory monitoring")
        except Exception as e:
            logger.debug(f"[PERFORMANCE] Error getting memory usage: {e}")
        
        # Log queue status
        queue_size = self.frame_queue.qsize()
        queue_utilization = (queue_size / self.frame_queue.maxsize) * 100
        
        logger.info(f"[PERFORMANCE] Queue: {queue_size}/{self.frame_queue.maxsize} "
                   f"({queue_utilization:.1f}% full)")
        
        # Log error statistics
        total_errors = sum(self.error_stats.values())
        if total_errors > 0:
            logger.info(f"[PERFORMANCE] Errors - Read: {self.error_stats['frame_read_errors']}, "
                       f"Processing: {self.error_stats['frame_processing_errors']}, "
                       f"Write: {self.error_stats['video_write_errors']}, "
                       f"Total: {total_errors}")
        
        # Log processing health
        current_time = time.time()
        time_since_last_frame = current_time - self.last_successful_frame_time
        logger.info(f"[PERFORMANCE] Health: {time_since_last_frame:.1f}s since last successful frame")

    def _log_processing_stage_transition(self, stage_name, status='start'):
        """Log processing stage transitions for detailed tracking"""
        current_time = time.time()
        
        if status == 'start':
            self.progress_tracking['processing_stages'][stage_name]['start'] = current_time
            logger.info(f"[STAGE] {stage_name.upper()} stage started")
        elif status == 'end':
            stage_info = self.progress_tracking['processing_stages'][stage_name]
            if stage_info['start']:
                duration = current_time - stage_info['start']
                stage_info['end'] = current_time
                stage_info['duration'] = duration
                logger.info(f"[STAGE] {stage_name.upper()} stage completed in {duration:.2f}s")
            else:
                logger.warning(f"[STAGE] {stage_name.upper()} stage ended without start time")

    def get_progress_info(self):
        """Get comprehensive progress information for API responses"""
        progress_info = self._calculate_accurate_progress()
        
        # Check for stalled processing
        current_time = time.time()
        time_since_heartbeat = current_time - self.last_heartbeat
        is_stalled = time_since_heartbeat > (self.heartbeat_interval * 3)  # 3x heartbeat interval
        
        if is_stalled and self.processing and not self.processing_complete:
            logger.warning(f"[STALL_DETECTION] Processing may be stalled - {time_since_heartbeat:.1f}s since last heartbeat")
            self.processing_stalled = True
        
        # Add additional context
        progress_info.update({
            'processing': self.processing,
            'processing_complete': self.processing_complete,
            'processing_stalled': self.processing_stalled,
            'time_since_heartbeat': time_since_heartbeat,
            'error_count': sum(self.error_stats.values()),
            'queue_size': self.frame_queue.qsize(),
            'queue_max': self.frame_queue.maxsize,
            'consecutive_failures': self.consecutive_read_failures,
            'performance_metrics': self.progress_tracking['performance_metrics'].copy(),
            'processing_stages': {
                stage: info.copy() for stage, info in self.progress_tracking['processing_stages'].items()
            }
        })
        
        return progress_info
    
    def _capture_partial_results(self):
        """
        Capture current processing results for partial saving when stopping manually
        
        Returns:
            dict: Partial results including current counts and processing state
        """
        try:
            logger.info("[PARTIAL_RESULTS] Capturing partial results for manual stop")
            
            # Get the latest processing result
            latest_result = self.get_latest_result()
            
            if not latest_result:
                logger.warning("[PARTIAL_RESULTS] No latest result available")
                return None
            
            # Create comprehensive partial results
            partial_results = {
                'in_count': latest_result.get('in_count', 0),
                'total_tracks': latest_result.get('total_tracks', 0),
                'classwise_counts': latest_result.get('classwise_counts', {}),
                'frames_processed': self.frames_processed,
                'total_frames': self.total_frames,
                'completion_percentage': (self.frames_processed / self.total_frames * 100) if self.total_frames > 0 else 0,
                'processing_fps': latest_result.get('fps', 0),
                'timestamp': time.time(),
                'stop_reason': 'manual_stop',
                'video_source': self.video_source,
                'model_name': self.model_name,
                'model_metadata': latest_result.get('model_metadata', {
                    'model_path': getattr(self.counter, 'model_path', None) if self.counter else None,
                    'fallback_used': getattr(self.counter, 'fallback_used', False) if self.counter else False,
                    'model_file_name': self.model_name
                }),
                'output_video_path': self.output_video_path
            }
            
            logger.info(f"[PARTIAL_RESULTS] Captured partial results - "
                       f"Frames: {partial_results['frames_processed']}/{partial_results['total_frames']}, "
                       f"Objects: {partial_results['in_count']}, "
                       f"Completion: {partial_results['completion_percentage']:.1f}%")
            
            return partial_results
            
        except Exception as e:
            logger.error(f"[PARTIAL_RESULTS] Error capturing partial results: {e}")
            traceback.print_exc()
            return None
    
    def _capture_processing_stats(self):
        """
        Capture comprehensive processing statistics for manual stop
        
        Returns:
            dict: Processing statistics including performance metrics and error counts
        """
        try:
            logger.info("[PROCESSING_STATS] Capturing processing statistics")
            
            current_time = time.time()
            total_processing_time = current_time - (self.progress_tracking['start_time'] or current_time)
            
            processing_stats = {
                'total_processing_time': total_processing_time,
                'frames_processed': self.frames_processed,
                'total_frames': self.total_frames,
                'average_fps': self.frames_processed / total_processing_time if total_processing_time > 0 else 0,
                'peak_fps': self.progress_tracking['performance_metrics']['peak_processing_fps'],
                'error_stats': self.error_stats.copy(),
                'total_errors': sum(self.error_stats.values()),
                'consecutive_read_failures': self.consecutive_read_failures,
                'queue_utilization': {
                    'current_size': self.frame_queue.qsize(),
                    'max_size': self.frame_queue.maxsize,
                    'utilization_percent': (self.frame_queue.qsize() / self.frame_queue.maxsize * 100) if self.frame_queue.maxsize > 0 else 0
                },
                'processing_stages': {
                    stage: info.copy() for stage, info in self.progress_tracking['processing_stages'].items()
                },
                'stop_requested_time': getattr(self, 'stop_requested_time', current_time),
                'manually_stopped': getattr(self, 'manually_stopped', False)
            }
            
            logger.info(f"[PROCESSING_STATS] Captured processing stats - "
                       f"Time: {total_processing_time:.2f}s, "
                       f"Avg FPS: {processing_stats['average_fps']:.2f}, "
                       f"Errors: {processing_stats['total_errors']}")
            
            return processing_stats
            
        except Exception as e:
            logger.error(f"[PROCESSING_STATS] Error capturing processing stats: {e}")
            traceback.print_exc()
            return None

