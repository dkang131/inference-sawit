"""
Video file validation utility for comprehensive video file validation and management.

This module provides functions to validate video files for format, size, integrity,
and other properties before processing. It supports the requirements for video
file validation as specified in the video processing pipeline.
"""

import os
import cv2
import logging
import mimetypes
import time
from typing import Dict, List, Tuple, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class VideoValidationError(Exception):
    """Custom exception for video validation errors"""
    pass

class VideoValidator:
    """
    Comprehensive video file validator that checks format, size, integrity,
    and other properties of video files before processing.
    """
    
    # Supported video formats based on requirements (1.1)
    SUPPORTED_FORMATS = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo', 
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska'
    }
    
    # File size limits (configurable)
    MAX_FILE_SIZE_MB = 500  # 500MB default limit
    MIN_FILE_SIZE_KB = 10   # 10KB minimum size
    
    # Video property limits
    MAX_DURATION_SECONDS = 3600  # 1 hour max
    MIN_DURATION_SECONDS = 1     # 1 second min
    MAX_RESOLUTION = (4096, 4096)  # 4K max
    MIN_RESOLUTION = (64, 64)      # 64x64 min
    MIN_FPS = 1
    MAX_FPS = 120
    
    def __init__(self, max_file_size_mb: int = None):
        """
        Initialize video validator with optional custom file size limit.
        
        Args:
            max_file_size_mb: Maximum file size in MB (optional)
        """
        if max_file_size_mb:
            self.MAX_FILE_SIZE_MB = max_file_size_mb
        
        logger.info(f"[VALIDATOR] Initialized with max file size: {self.MAX_FILE_SIZE_MB}MB")
    
    def validate_video_file(self, file_path: str) -> Dict[str, Union[bool, str, Dict]]:
        """
        Perform comprehensive validation of a video file.
        
        Args:
            file_path: Path to the video file to validate
            
        Returns:
            Dict containing validation results with the following structure:
            {
                'is_valid': bool,
                'error_message': str or None,
                'validation_details': {
                    'file_exists': bool,
                    'format_valid': bool,
                    'size_valid': bool,
                    'integrity_valid': bool,
                    'properties_valid': bool
                },
                'file_info': {
                    'size_mb': float,
                    'format': str,
                    'duration': float,
                    'resolution': tuple,
                    'fps': float,
                    'codec': str
                },
                'warnings': list
            }
        """
        logger.info(f"[VALIDATOR] Starting comprehensive validation for: {file_path}")
        
        validation_result = {
            'is_valid': False,
            'error_message': None,
            'validation_details': {
                'file_exists': False,
                'format_valid': False,
                'size_valid': False,
                'integrity_valid': False,
                'properties_valid': False
            },
            'file_info': {},
            'warnings': []
        }
        
        try:
            # Step 1: Check file existence
            if not self._validate_file_exists(file_path):
                validation_result['error_message'] = "Video file does not exist"
                return validation_result
            
            validation_result['validation_details']['file_exists'] = True
            
            # Step 2: Validate file format
            format_result = self._validate_file_format(file_path)
            if not format_result['valid']:
                validation_result['error_message'] = format_result['error']
                return validation_result
            
            validation_result['validation_details']['format_valid'] = True
            validation_result['file_info']['format'] = format_result['format']
            
            # Step 3: Validate file size
            size_result = self._validate_file_size(file_path)
            if not size_result['valid']:
                validation_result['error_message'] = size_result['error']
                return validation_result
            
            validation_result['validation_details']['size_valid'] = True
            validation_result['file_info']['size_mb'] = size_result['size_mb']
            
            # Step 4: Validate video integrity and properties
            integrity_result = self._validate_video_integrity(file_path)
            if not integrity_result['valid']:
                validation_result['error_message'] = integrity_result['error']
                return validation_result
            
            validation_result['validation_details']['integrity_valid'] = True
            validation_result['validation_details']['properties_valid'] = True
            
            # Add video properties to file info
            validation_result['file_info'].update(integrity_result['properties'])
            validation_result['warnings'].extend(integrity_result['warnings'])
            
            # All validations passed
            validation_result['is_valid'] = True
            logger.info(f"[VALIDATOR] Validation successful for: {file_path}")
            
        except Exception as e:
            logger.error(f"[VALIDATOR] Unexpected error during validation: {e}")
            validation_result['error_message'] = f"Validation failed: {str(e)}"
        
        return validation_result
    
    def _validate_file_exists(self, file_path: str) -> bool:
        """Check if file exists and is accessible."""
        try:
            return os.path.exists(file_path) and os.path.isfile(file_path)
        except Exception as e:
            logger.error(f"[VALIDATOR] Error checking file existence: {e}")
            return False
    
    def _validate_file_format(self, file_path: str) -> Dict[str, Union[bool, str]]:
        """
        Validate video file format against supported formats.
        
        Returns:
            Dict with 'valid', 'format', and 'error' keys
        """
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            
            if ext not in self.SUPPORTED_FORMATS:
                supported_formats = ', '.join(self.SUPPORTED_FORMATS.keys())
                return {
                    'valid': False,
                    'error': f"Unsupported video format '{ext}'. Supported formats: {supported_formats}",
                    'format': ext
                }
            
            # Verify MIME type if possible
            mime_type, _ = mimetypes.guess_type(file_path)
            expected_mime = self.SUPPORTED_FORMATS[ext]
            
            if mime_type and mime_type != expected_mime:
                logger.warning(f"[VALIDATOR] MIME type mismatch: expected {expected_mime}, got {mime_type}")
            
            return {
                'valid': True,
                'format': ext,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"[VALIDATOR] Error validating file format: {e}")
            return {
                'valid': False,
                'error': f"Error checking file format: {str(e)}",
                'format': None
            }
    
    def _validate_file_size(self, file_path: str) -> Dict[str, Union[bool, str, float]]:
        """
        Validate video file size against configured limits.
        
        Returns:
            Dict with 'valid', 'size_mb', and 'error' keys
        """
        try:
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            file_size_kb = file_size_bytes / 1024
            
            # Check minimum size
            if file_size_kb < self.MIN_FILE_SIZE_KB:
                return {
                    'valid': False,
                    'error': f"File too small: {file_size_kb:.1f}KB (minimum: {self.MIN_FILE_SIZE_KB}KB)",
                    'size_mb': file_size_mb
                }
            
            # Check maximum size
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                return {
                    'valid': False,
                    'error': f"File too large: {file_size_mb:.1f}MB (maximum: {self.MAX_FILE_SIZE_MB}MB)",
                    'size_mb': file_size_mb
                }
            
            return {
                'valid': True,
                'size_mb': file_size_mb,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"[VALIDATOR] Error validating file size: {e}")
            return {
                'valid': False,
                'error': f"Error checking file size: {str(e)}",
                'size_mb': 0
            }    

    def _validate_video_integrity(self, file_path: str) -> Dict[str, Union[bool, str, Dict, List]]:
        """
        Validate video file integrity and properties using OpenCV.
        
        Returns:
            Dict with 'valid', 'properties', 'warnings', and 'error' keys
        """
        cap = None
        try:
            # Open video file with OpenCV
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return {
                    'valid': False,
                    'error': "Cannot open video file - file may be corrupted or in unsupported format",
                    'properties': {},
                    'warnings': []
                }
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0
            
            # Get codec information (if available)
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)]) if fourcc else "unknown"
            
            properties = {
                'duration': duration,
                'resolution': (width, height),
                'fps': fps,
                'total_frames': total_frames,
                'codec': codec
            }
            
            warnings = []
            
            # Validate video properties
            validation_errors = []
            
            # Check duration
            if duration < self.MIN_DURATION_SECONDS:
                validation_errors.append(f"Video too short: {duration:.1f}s (minimum: {self.MIN_DURATION_SECONDS}s)")
            elif duration > self.MAX_DURATION_SECONDS:
                validation_errors.append(f"Video too long: {duration:.1f}s (maximum: {self.MAX_DURATION_SECONDS}s)")
            
            # Check resolution
            if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                validation_errors.append(f"Resolution too low: {width}x{height} (minimum: {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]})")
            elif width > self.MAX_RESOLUTION[0] or height > self.MAX_RESOLUTION[1]:
                validation_errors.append(f"Resolution too high: {width}x{height} (maximum: {self.MAX_RESOLUTION[0]}x{self.MAX_RESOLUTION[1]})")
            
            # Check FPS
            if fps < self.MIN_FPS:
                validation_errors.append(f"FPS too low: {fps} (minimum: {self.MIN_FPS})")
            elif fps > self.MAX_FPS:
                validation_errors.append(f"FPS too high: {fps} (maximum: {self.MAX_FPS})")
            
            # Check if we can read at least one frame
            ret, frame = cap.read()
            if not ret or frame is None:
                validation_errors.append("Cannot read video frames - file may be corrupted")
            else:
                # Add frame reading success to properties
                properties['first_frame_readable'] = True
                
                # Check for unusual properties that might indicate issues
                if total_frames <= 0:
                    warnings.append("Total frame count is zero or negative")
                if fps <= 0:
                    warnings.append("Invalid FPS detected")
                if width <= 0 or height <= 0:
                    warnings.append("Invalid video dimensions detected")
            
            # Return validation result
            if validation_errors:
                return {
                    'valid': False,
                    'error': "; ".join(validation_errors),
                    'properties': properties,
                    'warnings': warnings
                }
            
            return {
                'valid': True,
                'error': None,
                'properties': properties,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"[VALIDATOR] Error validating video integrity: {e}")
            return {
                'valid': False,
                'error': f"Error reading video file: {str(e)}",
                'properties': {},
                'warnings': []
            }
        finally:
            if cap:
                cap.release()

    def get_quick_validation_summary(self, file_path: str) -> str:
        """
        Get a quick validation summary for logging or user display.
        
        Args:
            file_path: Path to video file
            
        Returns:
            String summary of validation result
        """
        result = self.validate_video_file(file_path)
        
        if result['is_valid']:
            file_info = result['file_info']
            return (f"Valid video: {file_info.get('format', 'unknown')} format, "
                   f"{file_info.get('size_mb', 0):.1f}MB, "
                   f"{file_info.get('duration', 0):.1f}s duration, "
                   f"{file_info.get('resolution', (0,0))[0]}x{file_info.get('resolution', (0,0))[1]} resolution")
        else:
            return f"Invalid video: {result['error_message']}"


class VideoFileManager:
    """
    Video file management utility for cleanup and file operations.
    Handles automatic cleanup of temporary uploaded videos and manual removal.
    """
    
    def __init__(self, upload_dir: str = None, output_dir: str = None):
        """
        Initialize video file manager.
        
        Args:
            upload_dir: Directory for uploaded videos (optional)
            output_dir: Directory for processed videos (optional)
        """
        # Set default directories relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.upload_dir = upload_dir or os.path.join(project_root, 'uploads')
        self.output_dir = output_dir or os.path.join(project_root, 'outputs')
        
        # Ensure directories exist
        self._ensure_directories_exist()
        
        logger.info(f"[FILE_MANAGER] Initialized - Upload dir: {self.upload_dir}, Output dir: {self.output_dir}")
    
    def _ensure_directories_exist(self):
        """Ensure upload and output directories exist."""
        for directory in [self.upload_dir, self.output_dir]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"[FILE_MANAGER] Created directory: {directory}")
                except Exception as e:
                    logger.error(f"[FILE_MANAGER] Error creating directory {directory}: {e}")
    
    def cleanup_uploaded_video(self, video_path: str) -> Dict[str, Union[bool, str]]:
        """
        Remove an uploaded video file.
        
        Args:
            video_path: Path to the video file to remove
            
        Returns:
            Dict with 'success' and 'message' keys
        """
        try:
            if not os.path.exists(video_path):
                return {
                    'success': False,
                    'message': f"Video file not found: {video_path}"
                }
            
            # Security check: ensure file is within upload directory
            abs_video_path = os.path.abspath(video_path)
            abs_upload_dir = os.path.abspath(self.upload_dir)
            
            if not abs_video_path.startswith(abs_upload_dir):
                logger.warning(f"[FILE_MANAGER] Security violation: attempted to delete file outside upload directory: {video_path}")
                return {
                    'success': False,
                    'message': "Cannot delete file outside upload directory"
                }
            
            # Get file size for logging
            file_size = os.path.getsize(video_path)
            
            # Remove the file
            os.remove(video_path)
            
            logger.info(f"[FILE_MANAGER] Successfully removed video file: {video_path} ({file_size} bytes)")
            return {
                'success': True,
                'message': f"Video file removed successfully"
            }
            
        except Exception as e:
            logger.error(f"[FILE_MANAGER] Error removing video file {video_path}: {e}")
            return {
                'success': False,
                'message': f"Error removing video file: {str(e)}"
            }
    
    def cleanup_old_videos(self, max_age_hours: int = 24) -> Dict[str, Union[int, List]]:
        """
        Clean up old uploaded videos based on age.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup (default: 24)
            
        Returns:
            Dict with cleanup statistics
        """
        logger.info(f"[FILE_MANAGER] Starting cleanup of videos older than {max_age_hours} hours")
        
        cleanup_stats = {
            'files_checked': 0,
            'files_removed': 0,
            'files_failed': 0,
            'bytes_freed': 0,
            'removed_files': [],
            'failed_files': []
        }
        
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            if not os.path.exists(self.upload_dir):
                logger.warning(f"[FILE_MANAGER] Upload directory does not exist: {self.upload_dir}")
                return cleanup_stats
            
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                
                # Skip directories
                if not os.path.isfile(file_path):
                    continue
                
                cleanup_stats['files_checked'] += 1
                
                try:
                    # Check file age
                    file_mtime = os.path.getmtime(file_path)
                    file_age = current_time - file_mtime
                    
                    if file_age > max_age_seconds:
                        # File is old enough to be cleaned up
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        
                        cleanup_stats['files_removed'] += 1
                        cleanup_stats['bytes_freed'] += file_size
                        cleanup_stats['removed_files'].append(filename)
                        
                        logger.info(f"[FILE_MANAGER] Removed old video: {filename} (age: {file_age/3600:.1f}h, size: {file_size} bytes)")
                
                except Exception as e:
                    logger.error(f"[FILE_MANAGER] Error processing file {filename}: {e}")
                    cleanup_stats['files_failed'] += 1
                    cleanup_stats['failed_files'].append(filename)
            
            logger.info(f"[FILE_MANAGER] Cleanup completed - Removed: {cleanup_stats['files_removed']}, "
                       f"Failed: {cleanup_stats['files_failed']}, Freed: {cleanup_stats['bytes_freed']} bytes")
            
        except Exception as e:
            logger.error(f"[FILE_MANAGER] Error during cleanup: {e}")
        
        return cleanup_stats
    
    def get_upload_directory_info(self) -> Dict[str, Union[int, float, List]]:
        """
        Get information about the upload directory.
        
        Returns:
            Dict with directory statistics
        """
        try:
            if not os.path.exists(self.upload_dir):
                return {
                    'exists': False,
                    'file_count': 0,
                    'total_size_mb': 0,
                    'files': []
                }
            
            files = []
            total_size = 0
            
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_mtime = os.path.getmtime(file_path)
                    
                    files.append({
                        'name': filename,
                        'size_bytes': file_size,
                        'size_mb': file_size / (1024 * 1024),
                        'modified_time': file_mtime
                    })
                    
                    total_size += file_size
            
            return {
                'exists': True,
                'file_count': len(files),
                'total_size_mb': total_size / (1024 * 1024),
                'files': files
            }
            
        except Exception as e:
            logger.error(f"[FILE_MANAGER] Error getting directory info: {e}")
            return {
                'exists': False,
                'file_count': 0,
                'total_size_mb': 0,
                'files': []
            }


# Convenience functions for easy import and use
def validate_video_file(file_path: str, max_file_size_mb: int = None) -> Dict:
    """
    Convenience function to validate a video file.
    
    Args:
        file_path: Path to video file
        max_file_size_mb: Maximum file size in MB (optional)
        
    Returns:
        Validation result dictionary
    """
    validator = VideoValidator(max_file_size_mb)
    return validator.validate_video_file(file_path)

def cleanup_video_file(file_path: str, upload_dir: str = None) -> Dict:
    """
    Convenience function to cleanup a video file.
    
    Args:
        file_path: Path to video file
        upload_dir: Upload directory (optional)
        
    Returns:
        Cleanup result dictionary
    """
    manager = VideoFileManager(upload_dir)
    return manager.cleanup_uploaded_video(file_path)