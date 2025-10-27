from flask import Flask, render_template, request, jsonify, Response, send_file  # pyright: ignore[reportMissingImports]
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]
from datetime import datetime
import pytz
import logging
import cv2
import torch
import os
import json
import threading
import time
import base64
import numpy as np

from extensions import db
from models import Video
from utils.video_processor import VideoProcess
from utils.database_manager import DatabaseManager
from utils.video_validator import VideoValidator, VideoFileManager

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Global variable to store the video processor
video_processor = None
processing_thread = None
current_video_filename = None

# Initialize database manager
db_manager = DatabaseManager(max_retries=3, retry_delay=1.0)

# Initialize video file manager for cleanup operations
video_file_manager = VideoFileManager()

def perform_startup_cleanup():
    """Perform automatic cleanup of old videos on application startup"""
    try:
        logger.info("[STARTUP] Performing automatic cleanup of old videos")
        cleanup_stats = video_file_manager.cleanup_old_videos(max_age_hours=48)  # Clean up videos older than 48 hours
        
        if cleanup_stats['files_removed'] > 0:
            logger.info(f"[STARTUP] Startup cleanup completed - Removed {cleanup_stats['files_removed']} old videos, "
                       f"freed {cleanup_stats['bytes_freed']} bytes")
        else:
            logger.info("[STARTUP] Startup cleanup completed - No old videos to remove")
            
    except Exception as e:
        logger.error(f"[STARTUP] Error during startup cleanup: {e}")

# Perform startup cleanup when the module is loaded
perform_startup_cleanup()

def get_available_models():
    """Get list of available models from the model directory with validation"""
    logger.info("[MODELS] Scanning for available models")
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    models = []
    
    if not os.path.exists(model_dir):
        logger.warning(f"[MODELS] Model directory does not exist: {model_dir}")
        return ['No models available']
    
    try:
        # Import validation function
        from utils.counter import validate_model_file
        
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            
            # Check if it's a file with valid extension
            if os.path.isfile(item_path) and item.endswith(('.pt', '.pth', '.onnx')):
                # Validate the model file
                validation_result = validate_model_file(item_path)
                
                if validation_result['is_valid']:
                    models.append(item)
                    logger.debug(f"[MODELS] Valid model found: {item}")
                else:
                    logger.warning(f"[MODELS] Invalid model file {item}: {validation_result['error']}")
        
        # Sort models for consistent ordering
        models.sort()
        
        if not models:
            logger.warning("[MODELS] No valid models found in model directory")
            models = ['No models available']
        else:
            logger.info(f"[MODELS] Found {len(models)} valid models: {models}")
    
    except Exception as e:
        logger.error(f"[MODELS] Error scanning model directory: {e}")
        models = ['No models available']
    
    return models

@app.route('/')
def index():
    models = get_available_models()
    return render_template('index.html', models=models)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload with comprehensive validation"""
    global current_video_filename
    
    print("[UPLOAD] Starting video upload process")  # Print statement for immediate visibility
    logger.info("[UPLOAD] Starting video upload process")
    start_time = time.time()
    
    if 'video' not in request.files:
        print("[UPLOAD] No video file provided in request")  # Print statement for immediate visibility
        logger.error("[UPLOAD] No video file provided in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        print("[UPLOAD] No video file selected")  # Print statement for immediate visibility
        logger.error("[UPLOAD] No video file selected")
        return jsonify({'error': 'No video file selected'}), 400
    
    # Initialize video validator and file manager
    validator = VideoValidator(max_file_size_mb=500)  # 500MB limit
    file_manager = VideoFileManager()
    
    # Save the uploaded video temporarily for validation
    upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print(f"[UPLOAD] Created upload directory: {upload_dir}")  # Print statement for immediate visibility
        logger.info(f"[UPLOAD] Created upload directory: {upload_dir}")
    
    # Store the original filename
    current_video_filename = file.filename
    video_path = os.path.join(upload_dir, file.filename)
    
    try:
        # Save file temporarily
        file.save(video_path)
        save_time = time.time() - start_time
        print(f"[UPLOAD] Video saved temporarily: {video_path} (Save time: {save_time:.2f}s)")
        logger.info(f"[UPLOAD] Video saved temporarily: {video_path} (Save time: {save_time:.2f}s)")
        
        # Perform comprehensive video validation
        validation_start = time.time()
        validation_result = validator.validate_video_file(video_path)
        validation_time = time.time() - validation_start
        
        print(f"[UPLOAD] Video validation completed (Validation time: {validation_time:.2f}s)")
        logger.info(f"[UPLOAD] Video validation completed (Validation time: {validation_time:.2f}s)")
        
        if not validation_result['is_valid']:
            # Validation failed - remove the uploaded file and return error
            cleanup_result = file_manager.cleanup_uploaded_video(video_path)
            error_message = validation_result['error_message']
            
            print(f"[UPLOAD] Video validation failed: {error_message}")
            logger.error(f"[UPLOAD] Video validation failed: {error_message}")
            
            return jsonify({
                'error': f'Invalid video file: {error_message}',
                'validation_details': validation_result['validation_details'],
                'file_cleaned_up': cleanup_result['success']
            }), 400
        
        # Validation successful - log file information
        file_info = validation_result['file_info']
        warnings = validation_result['warnings']
        
        success_message = (f"Video uploaded and validated successfully: {file_info.get('format', 'unknown')} format, "
                          f"{file_info.get('size_mb', 0):.1f}MB, {file_info.get('duration', 0):.1f}s duration, "
                          f"{file_info.get('resolution', (0,0))[0]}x{file_info.get('resolution', (0,0))[1]} resolution")
        
        upload_time = time.time() - start_time
        print(f"[UPLOAD] {success_message} (Total upload time: {upload_time:.2f}s)")
        logger.info(f"[UPLOAD] {success_message} (Total upload time: {upload_time:.2f}s)")
        
        # Log any warnings
        if warnings:
            for warning in warnings:
                logger.warning(f"[UPLOAD] Video validation warning: {warning}")
        
        return jsonify({
            'message': 'Video uploaded and validated successfully',
            'video_path': video_path,
            'file_info': file_info,
            'validation_details': validation_result['validation_details'],
            'warnings': warnings
        }), 200
        
    except Exception as e:
        # Error occurred - try to clean up the file if it exists
        if os.path.exists(video_path):
            try:
                file_manager.cleanup_uploaded_video(video_path)
                logger.info(f"[UPLOAD] Cleaned up file after error: {video_path}")
            except Exception as cleanup_error:
                logger.error(f"[UPLOAD] Error during cleanup: {cleanup_error}")
        
        upload_time = time.time() - start_time
        print(f"[UPLOAD] Error during upload: {e} (Time: {upload_time:.2f}s)")
        logger.error(f"[UPLOAD] Error during upload: {e} (Time: {upload_time:.2f}s)")
        return jsonify({'error': f'Error processing video file: {str(e)}'}), 500

@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    """Start video processing with selected model"""
    global video_processor, processing_thread, current_video_filename
    
    logger.info("[START] Starting video processing request")
    start_time = time.time()
    
    data = request.json
    video_path = data.get('video_path')
    model_name = data.get('model_name')
    
    if not video_path or not model_name:
        logger.error("[START] Missing video_path or model_name in request")
        return jsonify({'error': 'Missing video_path or model_name'}), 400
    
    if not os.path.exists(video_path):
        logger.error(f"[START] Video file not found: {video_path}")
        return jsonify({'error': 'Video file not found'}), 404
    
    # Store the video filename for later use
    current_video_filename = os.path.basename(video_path)
    logger.info(f"[START] Processing video: {video_path} with model: {model_name}")
    
    # Stop any existing processing
    if video_processor and hasattr(video_processor, 'stop'):
        logger.info("[START] Stopping existing video processor")
        try:
            video_processor.stop()
        except Exception as e:
            logger.error(f"[START] Error stopping existing video processor: {e}")
    
    # Create new video processor
    try:
        video_processor = VideoProcess()
        logger.info("[START] VideoProcess instance created successfully")
    except Exception as e:
        logger.error(f"[START] Error creating VideoProcess instance: {e}")
        return jsonify({'error': f"Failed to create video processor: {e}"}), 500
    
    # Start processing directly (not in a separate thread since start_processing already creates threads)
    try:
        success, message = video_processor.start_processing(video_path, model_name)
        
        init_time = time.time() - start_time
        
        if success:
            logger.info(f"[START] Processing started successfully with video: {video_path} and model: {model_name} "
                       f"(Initialization time: {init_time:.2f}s)")
            return jsonify({'message': 'Processing started'}), 200
        else:
            logger.error(f"[START] Failed to start processing: {message}")
            return jsonify({'error': f'Failed to start processing: {message}'}), 500
    except Exception as e:
        logger.error(f"[START] Error starting processing: {e}")
        # Clean up on error
        if video_processor:
            try:
                video_processor.stop()
            except:
                pass
        return jsonify({'error': str(e)}), 500

@app.route('/api/processing_status')
def processing_status():
    """Get comprehensive processing status including progress, completion, and performance metrics"""
    global video_processor
    
    if not video_processor:
        logger.debug("[STATUS] No active video processor")
        return jsonify({
            'status': 'idle',
            'processing': False,
            'data': None,
            'progress': {
                'percentage': 0,
                'frames_processed': 0,
                'total_frames': 0,
                'stage': 'idle'
            },
            'performance': {
                'current_fps': 0,
                'avg_fps': 0,
                'processing_time': 0
            },
            'error_state': {
                'has_errors': False,
                'error_count': 0,
                'consecutive_failures': 0
            }
        }), 200
    
    try:
        logger.debug("[STATUS] Retrieving comprehensive processing status")
        
        # Get latest processing result
        result = video_processor.get_latest_result()
        
        # Get comprehensive progress information
        progress_info = video_processor.get_progress_info()
        
        # Get completion status
        completion_status = video_processor.get_completion_status()
        
        # Process latest result for JSON serialization
        processed_result = {}
        if result:
            for key, value in result.items():
                if key == 'frame':
                    continue  # Skip frame data for API response
                elif key == 'classwise_counts':
                    # Process classwise counts to ensure they are simple values, not objects
                    if isinstance(value, dict):
                        processed_counts = {}
                        for class_name, count_value in value.items():
                            # If count_value is a dict (like {'IN': 5, 'OUT': 3}), sum the values
                            if isinstance(count_value, dict):
                                # Extract numeric values and sum them
                                total_count = 0
                                for sub_key, sub_value in count_value.items():
                                    if isinstance(sub_value, (int, float)):
                                        total_count += sub_value
                                processed_counts[class_name] = total_count
                            else:
                                processed_counts[class_name] = count_value
                        processed_result[key] = processed_counts
                    else:
                        processed_result[key] = {}
                elif hasattr(value, 'tolist'):  # numpy arrays
                    processed_result[key] = value.tolist()
                else:
                    processed_result[key] = value
        
        # Check if processing is complete
        is_complete = video_processor.is_processing_complete()
        
        # Build comprehensive status response
        status_response = {
            'status': 'complete' if is_complete else 'processing',
            'processing': video_processor.processing,
            'data': processed_result,
            'progress': {
                'percentage': progress_info.get('percentage', 0),
                'frames_processed': progress_info.get('frames_processed', 0),
                'total_frames': progress_info.get('total_frames', 0),
                'frames_remaining': progress_info.get('frames_remaining', 0),
                'stage': progress_info.get('stage', 'unknown'),
                'estimated_time_remaining': progress_info.get('estimated_time_remaining', 0),
                'elapsed_time': progress_info.get('elapsed_time', 0)
            },
            'performance': {
                'current_fps': progress_info.get('current_fps', 0),
                'avg_fps': progress_info.get('avg_fps', 0),
                'peak_fps': progress_info.get('performance_metrics', {}).get('peak_processing_fps', 0),
                'processing_time': progress_info.get('elapsed_time', 0),
                'memory_usage_mb': progress_info.get('performance_metrics', {}).get('memory_usage_mb', 0)
            },
            'error_state': {
                'has_errors': progress_info.get('error_count', 0) > 0,
                'error_count': progress_info.get('error_count', 0),
                'consecutive_failures': progress_info.get('consecutive_failures', 0),
                'error_details': {
                    'frame_read_errors': video_processor.error_stats.get('frame_read_errors', 0),
                    'frame_processing_errors': video_processor.error_stats.get('frame_processing_errors', 0),
                    'video_write_errors': video_processor.error_stats.get('video_write_errors', 0)
                }
            },
            'queue_status': {
                'current_size': progress_info.get('queue_size', 0),
                'max_size': progress_info.get('queue_max', 0),
                'utilization_percent': (progress_info.get('queue_size', 0) / progress_info.get('queue_max', 1)) * 100 if progress_info.get('queue_max', 0) > 0 else 0
            },
            'completion_info': {
                'is_complete': is_complete,
                'completion_flags': completion_status.get('completion_flags', {}),
                'completion_method': completion_status.get('completion_status', {}).get('completion_method'),
                'completion_reason': completion_status.get('completion_status', {}).get('completion_reason')
            },
            'processing_stages': progress_info.get('processing_stages', {}),
            'timestamp': time.time()
        }
        
        # Log status at regular intervals with enhanced information
        frame_count = processed_result.get('frame_count', 0)
        if frame_count % 25 == 0:  # Log every 25 frames
            progress = status_response['progress']['percentage']
            current_fps = status_response['performance']['current_fps']
            error_count = status_response['error_state']['error_count']
            logger.info(f"[STATUS] Processing status - Frame: {frame_count}, Progress: {progress:.1f}%, "
                       f"FPS: {current_fps:.1f}, Errors: {error_count}, Complete: {is_complete}")
        
        # Auto-stop processing if complete
        if is_complete:
            logger.info("[STATUS] Auto-stopping processing as video has reached end")
            # We'll handle the stopping in a separate thread to avoid blocking
            import threading
            stop_thread = threading.Thread(target=auto_stop_processing)
            stop_thread.start()
        
        return jsonify(status_response), 200
        
    except Exception as e:
        logger.error(f"[STATUS] Error getting processing status: {e}")
        # Log the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        
        # Return error status with as much information as possible
        error_response = {
            'status': 'error',
            'processing': False,
            'error': str(e),
            'data': None,
            'progress': {
                'percentage': 0,
                'frames_processed': 0,
                'total_frames': 0,
                'stage': 'error'
            },
            'performance': {
                'current_fps': 0,
                'avg_fps': 0,
                'processing_time': 0
            },
            'error_state': {
                'has_errors': True,
                'error_count': 1,
                'consecutive_failures': 0,
                'api_error': str(e)
            },
            'timestamp': time.time()
        }
        
        return jsonify(error_response), 500

def auto_stop_processing():
    """Auto-stop processing when video reaches end"""
    global video_processor, processing_thread, current_video_filename
    
    # Small delay to ensure frontend has time to get the completion status
    time.sleep(1)
    
    if video_processor:
        logger.info("[AUTO_STOP] Starting auto-stop process")
        stop_start_time = time.time()
        
        # Get final results before stopping
        final_result = video_processor.get_latest_result()
        logger.info(f"[AUTO_STOP] Retrieved final result with frame count: {final_result.get('frame_count', 'N/A') if final_result else 'None'}")
        
        # Stop processing
        video_processor.stop()
        stop_time = time.time() - stop_start_time
        logger.info(f"[AUTO_STOP] Video processor stopped (Stop time: {stop_time:.2f}s)")
        
        # Save results to database using enhanced database manager
        if final_result:
            try:
                save_start_time = time.time()
                
                # Prepare processing metadata
                processing_metadata = {
                    'processing_fps': final_result.get('fps', 0),
                    'total_frames_processed': final_result.get('frame_count', 0),
                    'processing_duration': time.time() - stop_start_time if 'stop_start_time' in locals() else 0,
                    'completion_method': 'auto_complete'
                }
                
                # Use database manager to save results with retry logic and transaction management
                video_id = db_manager.save_video_results(
                    video_filename=current_video_filename or "unknown.mp4",
                    model_name=video_processor.model_name if hasattr(video_processor, 'model_name') else None,
                    processing_results=final_result,
                    processing_metadata=processing_metadata
                )
                
                save_time = time.time() - save_start_time
                if video_id:
                    logger.info(f"[AUTO_STOP] Results saved to database with ID {video_id}: {final_result.get('in_count', 0)} total objects "
                               f"(Save time: {save_time:.2f}s)")
                else:
                    logger.error(f"[AUTO_STOP] Failed to save results to database (Save time: {save_time:.2f}s)")
                    
            except Exception as e:
                logger.error(f"[AUTO_STOP] Unexpected error during database save: {e}")
                # Log the full traceback for debugging
                import traceback
                logger.error(traceback.format_exc())
        
        # Perform automatic cleanup of old videos after processing completes
        try:
            logger.info("[AUTO_STOP] Performing automatic cleanup after processing completion")
            cleanup_stats = video_file_manager.cleanup_old_videos(max_age_hours=24)
            
            if cleanup_stats['files_removed'] > 0:
                logger.info(f"[AUTO_STOP] Post-processing cleanup completed - Removed {cleanup_stats['files_removed']} old videos, "
                           f"freed {cleanup_stats['bytes_freed']} bytes")
        except Exception as e:
            logger.error(f"[AUTO_STOP] Error during post-processing cleanup: {e}")
        
        video_processor = None
        current_video_filename = None
        
        if processing_thread and processing_thread.is_alive():
            join_start_time = time.time()
            processing_thread.join(timeout=5.0)
            join_time = time.time() - join_start_time
            logger.info(f"[AUTO_STOP] Processing thread joined (Join time: {join_time:.2f}s)")
        
        logger.info("[AUTO_STOP] Auto-stop process completed")

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route"""
    logger.info("[VIDEO_FEED] Starting video feed stream")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Generate video frames for streaming"""
    global video_processor
    
    logger.info("[FRAME_GEN] Starting frame generation")
    start_time = time.time()
    frame_count = 0
    
    # Keep track of last frame to avoid sending duplicate frames
    last_frame_count = -1
    
    while True:
        if video_processor:
            result = video_processor.get_latest_result()
            if result and 'frame' in result:
                frame = result['frame']
                current_frame_count = result.get('frame_count', 0)
                
                # Only send frame if it's different from the last one
                if current_frame_count != last_frame_count:
                    frame_gen_start = time.time()
                    try:
                        if hasattr(frame, 'shape') and len(frame.shape) >= 2:
                            # Encode frame with consistent quality settings
                            ret, buffer = cv2.imencode('.jpg', frame)  # pyright: ignore[reportAttributeAccessIssue]
                            if ret:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                                last_frame_count = current_frame_count
                                frame_count += 1
                                
                                # Log frame generation performance
                                if frame_count % 50 == 0:
                                    elapsed = time.time() - start_time
                                    fps = frame_count / elapsed if elapsed > 0 else 0
                                    encode_time = time.time() - frame_gen_start
                                    logger.debug(f"[FRAME_GEN] Generated frame {frame_count} (FPS: {fps:.2f}, Encode time: {encode_time:.4f}s)")
                            else:
                                logger.warning("[FRAME_GEN] Failed to encode frame")
                        else:
                            logger.warning(f"[FRAME_GEN] Invalid frame data: {type(frame)}")
                    except Exception as e:
                        logger.error(f"[FRAME_GEN] Error encoding frame: {e}")
                
            # Check if processing is complete
            if video_processor.is_processing_complete():
                logger.info("[FRAME_GEN] Processing complete, stopping video feed")
                # Send a final frame to indicate completion
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Processing Complete", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # pyright: ignore[reportAttributeAccessIssue]
                ret, buffer = cv2.imencode('.jpg', blank_frame)  # pyright: ignore[reportAttributeAccessIssue]
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                break
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.033)  # ~30 FPS
    
    total_time = time.time() - start_time
    logger.info(f"[FRAME_GEN] Frame generation stopped. Total frames: {frame_count}, Total time: {total_time:.2f}s")

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """Stop video processing and save partial results to database with enhanced feedback"""
    global video_processor, processing_thread, current_video_filename
    
    logger.info("[STOP] Manual stop processing request received")
    stop_start_time = time.time()
    
    if not video_processor:
        logger.info("[STOP] No active processing to stop")
        return jsonify({
            'success': True,
            'message': 'No active processing to stop',
            'partial_results': None,
            'processing_stats': None
        }), 200
    
    # Get current processing state before stopping
    current_result = video_processor.get_latest_result()
    current_progress = video_processor.get_progress_info()
    
    logger.info(f"[STOP] Current processing state - Frame: {current_result.get('frame_count', 'N/A') if current_result else 'None'}, "
               f"Progress: {current_progress.get('percentage', 0):.1f}%")
    
    # Stop processing with partial result saving
    stop_result = video_processor.stop(save_partial_results=True)
    stop_time = time.time() - stop_start_time
    
    logger.info(f"[STOP] Video processor stopped (Stop time: {stop_time:.2f}s)")
    logger.info(f"[STOP] Stop result: {stop_result['message']}")
    
    # Prepare response data
    response_data = {
        'success': stop_result['success'],
        'message': stop_result['message'],
        'stop_time': stop_time,
        'frames_processed': stop_result.get('frames_processed', 0),
        'total_frames': stop_result.get('total_frames', 0),
        'completion_percentage': stop_result.get('completion_percentage', 0),
        'manually_stopped': True,
        'database_saved': False,
        'video_id': None
    }
    
    # Save partial results to database using enhanced database manager
    partial_results = stop_result.get('partial_results')
    if partial_results:
        try:
            save_start_time = time.time()
            
            # Prepare processing metadata with stop information
            processing_metadata = {
                'processing_fps': partial_results.get('processing_fps', 0),
                'total_frames_processed': partial_results.get('frames_processed', 0),
                'processing_duration': stop_result.get('processing_stats', {}).get('total_processing_time', 0),
                'completion_method': 'manual_stop',
                'completion_percentage': partial_results.get('completion_percentage', 0),
                'stop_time': stop_time,
                'manually_stopped': True,
                'partial_results': True
            }
            
            # Create a result object compatible with database manager
            database_result = {
                'in_count': partial_results.get('in_count', 0),
                'total_tracks': partial_results.get('total_tracks', 0),
                'classwise_counts': partial_results.get('classwise_counts', {}),
                'fps': partial_results.get('processing_fps', 0),
                'frame_count': partial_results.get('frames_processed', 0),
                'timestamp': partial_results.get('timestamp', time.time())
            }
            
            # Use database manager to save results with retry logic and transaction management
            video_id = db_manager.save_video_results(
                video_filename=current_video_filename or "unknown.mp4",
                model_name=partial_results.get('model_name') or (video_processor.model_name if hasattr(video_processor, 'model_name') else None),
                processing_results=database_result,
                processing_metadata=processing_metadata
            )
            
            save_time = time.time() - save_start_time
            if video_id:
                logger.info(f"[STOP] Partial results saved to database with ID {video_id}: {partial_results.get('in_count', 0)} total objects "
                           f"(Save time: {save_time:.2f}s, {partial_results.get('completion_percentage', 0):.1f}% complete)")
                response_data.update({
                    'database_saved': True,
                    'video_id': video_id,
                    'save_time': save_time,
                    'objects_detected': partial_results.get('in_count', 0)
                })
            else:
                logger.error(f"[STOP] Failed to save partial results to database (Save time: {save_time:.2f}s)")
                response_data['message'] += ' (Warning: Failed to save to database)'
                
        except Exception as e:
            logger.error(f"[STOP] Unexpected error during database save: {e}")
            # Log the full traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            response_data['message'] += f' (Warning: Database save error: {str(e)})'
    else:
        logger.warning("[STOP] No partial results available to save")
        response_data['message'] += ' (Warning: No partial results to save)'
    
    # Perform automatic cleanup after manual stop
    try:
        logger.info("[STOP] Performing automatic cleanup after manual stop")
        cleanup_stats = video_file_manager.cleanup_old_videos(max_age_hours=24)
        
        if cleanup_stats['files_removed'] > 0:
            logger.info(f"[STOP] Post-stop cleanup completed - Removed {cleanup_stats['files_removed']} old videos, "
                       f"freed {cleanup_stats['bytes_freed']} bytes")
    except Exception as e:
        logger.error(f"[STOP] Error during post-stop cleanup: {e}")
    
    # Clean up global variables
    video_processor = None
    current_video_filename = None
    
    # Wait for processing thread to finish
    if processing_thread and processing_thread.is_alive():
        join_start_time = time.time()
        processing_thread.join(timeout=5.0)
        join_time = time.time() - join_start_time
        logger.info(f"[STOP] Processing thread joined (Join time: {join_time:.2f}s)")
        if processing_thread.is_alive():
            logger.warning("[STOP] Processing thread did not finish within timeout")
            response_data['message'] += ' (Warning: Thread cleanup timeout)'
    
    logger.info(f"[STOP] Manual stop processing completed - {response_data['frames_processed']}/{response_data['total_frames']} frames processed")
    
    return jsonify(response_data), 200

@app.route('/api/results')
def get_results():
    """Get processing results from database"""
    logger.info("[RESULTS] Retrieving processing results from database")
    start_time = time.time()
    
    try:
        videos = Video.query.order_by(Video.created_at.desc()).limit(10).all()
        results = []
        for video in videos:
            results.append({
                'id': video.id,
                'video_file_name': video.video_file_name,
                'model_name': video.model_name,
                'total_count': video.total_count,
                'empty_count': video.empty_count,
                'empty_ls_count': video.empty_ls_count,
                'hb_count': video.hb_count,
                'hb_ls_count': video.hb_ls_count,
                'hb_ab_count': video.hb_ab_count,
                'unripe_count': video.unripe_count,
                'unripe_ls_count': video.unripe_ls_count,
                'unripe_ab_count': video.unripe_ab_count,
                'ripe_count': video.ripe_count,
                'ripe_ls_count': video.ripe_ls_count,
                'ripe_ab_count': video.ripe_ab_count,
                'overripe_count': video.overripe_count,
                'overripe_ls_count': video.overripe_ls_count,
                'overripe_ab_count': video.overripe_ab_count,
                'bunch_count': video.bunch_count,
                'created_at': video.created_at.isoformat() if video.created_at else None
            })
        
        query_time = time.time() - start_time
        logger.info(f"[RESULTS] Retrieved {len(results)} results from database (Query time: {query_time:.2f}s)")
        return jsonify({'results': results}), 200
    except Exception as e:
        logger.error(f"[RESULTS] Error getting results: {e}")
        query_time = time.time() - start_time
        logger.error(f"[RESULTS] Query failed after {query_time:.2f}s")
        return jsonify({'results': []}), 200

@app.route('/api/remove_video', methods=['POST'])
def remove_video():
    """Remove uploaded video file with enhanced validation and cleanup"""
    global current_video_filename
    
    logger.info("[REMOVE] Starting video removal process")
    start_time = time.time()
    
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path:
        logger.error("[REMOVE] Missing video_path in request")
        return jsonify({'error': 'Missing video_path'}), 400
    
    # Initialize file manager for secure cleanup
    file_manager = VideoFileManager()
    
    try:
        # Use file manager for secure removal
        cleanup_result = file_manager.cleanup_uploaded_video(video_path)
        remove_time = time.time() - start_time
        
        if cleanup_result['success']:
            logger.info(f"[REMOVE] Video removed successfully: {video_path} (Remove time: {remove_time:.2f}s)")
            
            # If this was the current video, clear the filename
            if current_video_filename and current_video_filename in video_path:
                current_video_filename = None
                logger.info("[REMOVE] Cleared current video filename")
            
            return jsonify({
                'message': 'Video removed successfully',
                'cleanup_result': cleanup_result
            }), 200
        else:
            logger.warning(f"[REMOVE] Failed to remove video: {cleanup_result['message']}")
            return jsonify({
                'error': cleanup_result['message'],
                'cleanup_result': cleanup_result
            }), 404 if 'not found' in cleanup_result['message'].lower() else 500
            
    except Exception as e:
        remove_time = time.time() - start_time
        logger.error(f"[REMOVE] Unexpected error during removal: {e} (Time: {remove_time:.2f}s)")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/cleanup_old_videos', methods=['POST'])
def cleanup_old_videos():
    """Clean up old uploaded videos based on age"""
    logger.info("[CLEANUP] Starting automatic cleanup of old videos")
    start_time = time.time()
    
    try:
        # Get max age from request (default 24 hours)
        data = request.json or {}
        max_age_hours = data.get('max_age_hours', 24)
        
        # Validate max_age_hours
        if not isinstance(max_age_hours, (int, float)) or max_age_hours <= 0:
            return jsonify({'error': 'Invalid max_age_hours parameter'}), 400
        
        # Initialize file manager and perform cleanup
        file_manager = VideoFileManager()
        cleanup_stats = file_manager.cleanup_old_videos(max_age_hours)
        
        cleanup_time = time.time() - start_time
        logger.info(f"[CLEANUP] Cleanup completed in {cleanup_time:.2f}s - "
                   f"Removed: {cleanup_stats['files_removed']}, "
                   f"Failed: {cleanup_stats['files_failed']}, "
                   f"Freed: {cleanup_stats['bytes_freed']} bytes")
        
        return jsonify({
            'message': f'Cleanup completed successfully',
            'cleanup_stats': cleanup_stats,
            'cleanup_time': cleanup_time
        }), 200
        
    except Exception as e:
        cleanup_time = time.time() - start_time
        logger.error(f"[CLEANUP] Error during cleanup: {e} (Time: {cleanup_time:.2f}s)")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.route('/api/upload_directory_info')
def get_upload_directory_info():
    """Get information about uploaded videos in the upload directory"""
    logger.info("[UPLOAD_INFO] Getting upload directory information")
    start_time = time.time()
    
    try:
        file_manager = VideoFileManager()
        directory_info = file_manager.get_upload_directory_info()
        
        info_time = time.time() - start_time
        logger.info(f"[UPLOAD_INFO] Directory info retrieved in {info_time:.2f}s - "
                   f"Files: {directory_info['file_count']}, "
                   f"Total size: {directory_info['total_size_mb']:.1f}MB")
        
        return jsonify({
            'directory_info': directory_info,
            'query_time': info_time
        }), 200
        
    except Exception as e:
        info_time = time.time() - start_time
        logger.error(f"[UPLOAD_INFO] Error getting directory info: {e} (Time: {info_time:.2f}s)")
        return jsonify({'error': f'Error getting directory info: {str(e)}'}), 500

@app.route('/api/health_check')
def health_check():
    """Health check endpoint for monitoring system status and processing health"""
    global video_processor, processing_thread
    
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'system': {
                'video_processor_active': video_processor is not None,
                'processing_thread_alive': processing_thread is not None and processing_thread.is_alive() if processing_thread else False,
                'processing_active': video_processor.processing if video_processor else False
            },
            'processing_health': {
                'is_healthy': True,
                'issues': []
            }
        }
        
        if video_processor:
            # Check processing health
            progress_info = video_processor.get_progress_info()
            error_count = progress_info.get('error_count', 0)
            consecutive_failures = progress_info.get('consecutive_failures', 0)
            
            # Determine health issues
            issues = []
            if error_count > 100:
                issues.append(f"High error count: {error_count}")
            if consecutive_failures > 50:
                issues.append(f"High consecutive failures: {consecutive_failures}")
            if progress_info.get('queue_size', 0) >= progress_info.get('queue_max', 1):
                issues.append("Frame queue is full")
            
            # Check if processing has stalled
            current_fps = progress_info.get('current_fps', 0)
            if video_processor.processing and current_fps < 0.1:
                issues.append("Processing appears to be stalled (very low FPS)")
            
            health_status['processing_health'] = {
                'is_healthy': len(issues) == 0,
                'issues': issues,
                'error_count': error_count,
                'consecutive_failures': consecutive_failures,
                'current_fps': current_fps,
                'queue_utilization': (progress_info.get('queue_size', 0) / progress_info.get('queue_max', 1)) * 100 if progress_info.get('queue_max', 0) > 0 else 0
            }
            
            if len(issues) > 0:
                health_status['status'] = 'degraded'
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"[HEALTH] Error during health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time(),
            'system': {
                'video_processor_active': False,
                'processing_thread_alive': False,
                'processing_active': False
            },
            'processing_health': {
                'is_healthy': False,
                'issues': [f"Health check failed: {str(e)}"]
            }
        }), 500

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get detailed performance metrics for monitoring processing efficiency"""
    global video_processor
    
    if not video_processor:
        return jsonify({
            'status': 'idle',
            'metrics': {
                'processing_fps': 0,
                'avg_fps': 0,
                'peak_fps': 0,
                'processing_time': 0,
                'memory_usage_mb': 0,
                'queue_utilization': 0,
                'error_rate': 0
            }
        }), 200
    
    try:
        logger.debug("[PERFORMANCE] Retrieving performance metrics")
        
        # Get comprehensive progress information
        progress_info = video_processor.get_progress_info()
        
        # Calculate error rate
        total_frames = progress_info.get('frames_processed', 0)
        total_errors = progress_info.get('error_count', 0)
        error_rate = (total_errors / total_frames * 100) if total_frames > 0 else 0
        
        # Get queue utilization
        queue_size = progress_info.get('queue_size', 0)
        queue_max = progress_info.get('queue_max', 1)
        queue_utilization = (queue_size / queue_max * 100) if queue_max > 0 else 0
        
        performance_metrics = {
            'status': 'processing' if video_processor.processing else 'idle',
            'metrics': {
                'processing_fps': progress_info.get('current_fps', 0),
                'avg_fps': progress_info.get('avg_fps', 0),
                'peak_fps': progress_info.get('performance_metrics', {}).get('peak_processing_fps', 0),
                'processing_time': progress_info.get('elapsed_time', 0),
                'memory_usage_mb': progress_info.get('performance_metrics', {}).get('memory_usage_mb', 0),
                'queue_utilization': round(queue_utilization, 2),
                'error_rate': round(error_rate, 2),
                'frames_processed': progress_info.get('frames_processed', 0),
                'total_frames': progress_info.get('total_frames', 0),
                'consecutive_failures': progress_info.get('consecutive_failures', 0)
            },
            'detailed_errors': {
                'frame_read_errors': video_processor.error_stats.get('frame_read_errors', 0),
                'frame_processing_errors': video_processor.error_stats.get('frame_processing_errors', 0),
                'video_write_errors': video_processor.error_stats.get('video_write_errors', 0),
                'recovery_attempts': video_processor.error_stats.get('recovery_attempts', 0),
                'successful_recoveries': video_processor.error_stats.get('successful_recoveries', 0)
            },
            'processing_stages': progress_info.get('processing_stages', {}),
            'timestamp': time.time()
        }
        
        return jsonify(performance_metrics), 200
        
    except Exception as e:
        logger.error(f"[PERFORMANCE] Error getting performance metrics: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'metrics': {
                'processing_fps': 0,
                'avg_fps': 0,
                'peak_fps': 0,
                'processing_time': 0,
                'memory_usage_mb': 0,
                'queue_utilization': 0,
                'error_rate': 0
            }
        }), 500

@app.route('/api/output_video/<int:video_id>')
def get_output_video(video_id):
    """Serve processed output video with comprehensive validation and error handling"""
    logger.info(f"[OUTPUT_VIDEO] Request for output video ID: {video_id}")
    start_time = time.time()
    
    try:
        # Validate video_id parameter
        if not isinstance(video_id, int) or video_id <= 0:
            logger.error(f"[OUTPUT_VIDEO] Invalid video ID: {video_id}")
            return jsonify({'error': 'Invalid video ID'}), 400
        
        # Query video record from database
        video = Video.query.get(video_id)
        if not video:
            logger.error(f"[OUTPUT_VIDEO] Video record not found for ID: {video_id}")
            return jsonify({'error': 'Video record not found'}), 404
        
        # Construct output video path based on video filename and model name
        if not video.video_file_name or not video.model_name:
            logger.error(f"[OUTPUT_VIDEO] Missing video filename or model name for video ID: {video_id}")
            return jsonify({'error': 'Insufficient metadata to locate processed video'}), 404
        
        # Construct expected output path
        outputs_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))
        
        # Generate expected output filename
        input_filename = video.video_file_name
        name, ext = os.path.splitext(input_filename)
        safe_model_name = video.model_name.replace('.', '_').replace('/', '_').replace('\\', '_')
        output_filename = f"{name}_processed_{safe_model_name}{ext}"
        output_path = os.path.join(outputs_dir, output_filename)
        
        if not output_path.startswith(outputs_dir):
            logger.error(f"[OUTPUT_VIDEO] Security violation: path outside outputs directory: {output_path}")
            return jsonify({'error': 'Invalid file path'}), 403
        
        # Check if output video file exists on filesystem
        if not os.path.exists(output_path):
            logger.error(f"[OUTPUT_VIDEO] Output video file not found: {output_path}")
            return jsonify({'error': 'Processed video file not found on server'}), 404
        
        # Validate file size and type
        try:
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                logger.error(f"[OUTPUT_VIDEO] Output video file is empty: {output_path}")
                return jsonify({'error': 'Processed video file is empty'}), 404
            
            # Check file extension
            _, ext = os.path.splitext(output_path)
            allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            if ext.lower() not in allowed_extensions:
                logger.warning(f"[OUTPUT_VIDEO] Unusual file extension: {ext} for file: {output_path}")
            
            logger.info(f"[OUTPUT_VIDEO] File validation passed - Size: {file_size} bytes, Extension: {ext}")
            
        except OSError as e:
            logger.error(f"[OUTPUT_VIDEO] Error accessing file {output_path}: {e}")
            return jsonify({'error': 'Error accessing processed video file'}), 500
        
        # Generate appropriate filename for download
        original_filename = video.video_file_name or "video"
        base_name = os.path.splitext(original_filename)[0]
        model_name = video.model_name or "default"
        safe_model_name = model_name.replace('.', '_').replace('/', '_').replace('\\', '_')
        download_filename = f"{base_name}_processed_{safe_model_name}{ext}"
        
        send_time = time.time() - start_time
        logger.info(f"[OUTPUT_VIDEO] Serving output video: {output_path} as {download_filename} "
                   f"(Lookup time: {send_time:.2f}s, File size: {file_size} bytes)")
        
        # Serve file with proper headers
        return send_file(
            output_path, 
            as_attachment=True,
            download_name=download_filename,
            mimetype='video/mp4'  # Default mimetype, browser will handle correctly
        )
        
    except Exception as e:
        send_time = time.time() - start_time
        logger.error(f"[OUTPUT_VIDEO] Unexpected error serving output video: {e} (Time: {send_time:.2f}s)")
        # Log full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error while serving video'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4786, host='0.0.0.0')