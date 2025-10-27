"""
Database manager utility for handling video processing results with enhanced error handling.
Provides transaction management, retry logic, and proper handling of nested count dictionaries.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Union
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy import text
from extensions import db
from models import Video

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Enhanced database manager for video processing results with robust error handling.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize database manager with retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts for database operations
            retry_delay: Base delay between retry attempts in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_stats = {
            'total_attempts': 0,
            'successful_saves': 0,
            'failed_saves': 0,
            'retry_attempts': 0,
            'connection_errors': 0,
            'integrity_errors': 0,
            'other_errors': 0
        }
    
    def save_video_results(self, 
                          video_filename: str,
                          model_name: Optional[str],
                          processing_results: Dict[str, Any],
                          processing_metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Save video processing results to database with transaction management and retry logic.
        
        Args:
            video_filename: Original video filename
            model_name: Name of the model used for processing
            processing_results: Dictionary containing processing results with counts
            processing_metadata: Optional dictionary with additional processing metadata
            
        Returns:
            Video record ID if successful, None if failed
        """
        self.error_stats['total_attempts'] += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"[DB_SAVE] Attempting to save results (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Extract and process counts from results
                processed_counts = self._extract_and_process_counts(processing_results)
                
                # Validate and prepare metadata
                validated_metadata = self._validate_processing_metadata(
                    video_filename, model_name, processing_metadata
                )
                
                # Create video record with transaction
                video_id = self._create_video_record_with_transaction(
                    video_file_name=validated_metadata['video_file_name'],
                    model_name=validated_metadata['model_name'],
                    total_count=processing_results.get('in_count', 0),
                    **processed_counts
                )
                
                if video_id:
                    self.error_stats['successful_saves'] += 1
                    logger.info(f"[DB_SAVE] Successfully saved video results with ID: {video_id}")
                    return video_id
                
            except OperationalError as e:
                self.error_stats['connection_errors'] += 1
                logger.warning(f"[DB_SAVE] Database connection error (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    self.error_stats['retry_attempts'] += 1
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"[DB_SAVE] Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"[DB_SAVE] Max retries exceeded for connection errors")
                    
            except IntegrityError as e:
                self.error_stats['integrity_errors'] += 1
                logger.error(f"[DB_SAVE] Database integrity error: {e}")
                # Don't retry integrity errors as they indicate data issues
                break
                
            except SQLAlchemyError as e:
                self.error_stats['other_errors'] += 1
                logger.error(f"[DB_SAVE] SQLAlchemy error (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    self.error_stats['retry_attempts'] += 1
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"[DB_SAVE] Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"[DB_SAVE] Max retries exceeded for SQLAlchemy errors")
                    
            except Exception as e:
                self.error_stats['other_errors'] += 1
                logger.error(f"[DB_SAVE] Unexpected error (attempt {attempt + 1}): {e}")
                logger.error(traceback.format_exc())
                
                if attempt < self.max_retries:
                    self.error_stats['retry_attempts'] += 1
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"[DB_SAVE] Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"[DB_SAVE] Max retries exceeded for unexpected errors")
        
        self.error_stats['failed_saves'] += 1
        logger.error(f"[DB_SAVE] Failed to save video results after {self.max_retries + 1} attempts")
        return None
    
    def _extract_and_process_counts(self, processing_results: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract and process count data from processing results, handling nested dictionaries.
        
        Args:
            processing_results: Raw processing results dictionary
            
        Returns:
            Dictionary with processed count values
        """
        classwise_counts = processing_results.get('classwise_counts', {})
        processed_counts = {}
        
        # Define count field mappings
        count_mappings = {
            'empty_count': 'empty',
            'empty_ls_count': 'empty-ls',
            'hb_count': 'hb',
            'hb_ls_count': 'hb-ls',
            'hb_ab_count': 'hb-ab',
            'unripe_count': 'unripe',
            'unripe_ls_count': 'unripe-ls',
            'unripe_ab_count': 'unripe-ab',
            'ripe_count': 'ripe',
            'ripe_ls_count': 'ripe-ls',
            'ripe_ab_count': 'ripe-ab',
            'overripe_count': 'overripe',
            'overripe_ls_count': 'overripe-ls',
            'overripe_ab_count': 'overripe-ab',
            'bunch_count': 'bunch'
        }
        
        # Process each count field
        for db_field, class_key in count_mappings.items():
            count_value = classwise_counts.get(class_key, 0)
            processed_counts[db_field] = self._extract_count_value(count_value, class_key)
        
        logger.debug(f"[DB_SAVE] Processed counts: {processed_counts}")
        return processed_counts
    
    def _validate_processing_metadata(self, 
                                    video_filename: str,
                                    model_name: Optional[str],
                                    processing_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and prepare processing metadata for database storage.
        
        Args:
            video_filename: Original video filename
            model_name: Name of the model used for processing
            processing_metadata: Optional additional metadata
            
        Returns:
            Dictionary with validated metadata
        """
        validated = {
            'video_file_name': video_filename or "unknown.mp4",
            'model_name': model_name
        }
        
        # Validate video filename
        if not validated['video_file_name'] or validated['video_file_name'].strip() == "":
            validated['video_file_name'] = "unknown.mp4"
            logger.warning("[DB_SAVE] Empty video filename, using default")
        
        # Validate model name
        if validated['model_name']:
            # Sanitize model name for database storage
            validated['model_name'] = str(validated['model_name']).strip()
            if len(validated['model_name']) > 100:  # Assuming 100 char limit
                validated['model_name'] = validated['model_name'][:100]
                logger.warning(f"[DB_SAVE] Model name truncated to 100 characters")
        
        # Process additional metadata if provided
        if processing_metadata:
            logger.debug(f"[DB_SAVE] Additional processing metadata: {processing_metadata}")
            # Future: Could extend Video model to store additional metadata as JSON
        
        logger.info(f"[DB_SAVE] Validated metadata - File: {validated['video_file_name']}, "
                   f"Model: {validated['model_name']}")
        
        return validated
    
    def _extract_count_value(self, count_value: Union[int, Dict[str, Any]], class_key: str) -> int:
        """
        Extract numeric count value from potentially nested dictionary structure.
        
        Args:
            count_value: Count value (int or dict with 'IN'/'OUT' keys)
            class_key: Class key for logging purposes
            
        Returns:
            Extracted integer count value
        """
        if isinstance(count_value, dict):
            # Handle nested count dictionaries like {'IN': 5, 'OUT': 3}
            if 'IN' in count_value:
                extracted_value = count_value['IN']
                logger.debug(f"[DB_SAVE] Extracted 'IN' count for {class_key}: {extracted_value}")
                return int(extracted_value) if isinstance(extracted_value, (int, float)) else 0
            elif 'OUT' in count_value:
                extracted_value = count_value['OUT']
                logger.debug(f"[DB_SAVE] Extracted 'OUT' count for {class_key}: {extracted_value}")
                return int(extracted_value) if isinstance(extracted_value, (int, float)) else 0
            else:
                # Sum all numeric values in the dictionary
                total = 0
                for key, value in count_value.items():
                    if isinstance(value, (int, float)):
                        total += value
                logger.debug(f"[DB_SAVE] Summed all values for {class_key}: {total}")
                return int(total)
        elif isinstance(count_value, (int, float)):
            return int(count_value)
        else:
            logger.warning(f"[DB_SAVE] Unexpected count value type for {class_key}: {type(count_value)}")
            return 0
    
    def _create_video_record_with_transaction(self, **kwargs) -> Optional[int]:
        """
        Create video record within a database transaction.
        
        Args:
            **kwargs: Video record fields
            
        Returns:
            Video record ID if successful, None if failed
        """
        try:
            # Start transaction
            logger.debug("[DB_SAVE] Starting database transaction")
            
            # Create video record
            video_record = Video(**kwargs)
            
            # Add to session and commit within transaction
            db.session.add(video_record)
            db.session.commit()
            
            video_id = video_record.id
            logger.info(f"[DB_SAVE] Transaction committed successfully, video ID: {video_id}")
            return video_id
            
        except Exception as e:
            # Rollback transaction on any error
            logger.error(f"[DB_SAVE] Transaction failed, rolling back: {e}")
            try:
                db.session.rollback()
                logger.info("[DB_SAVE] Transaction rolled back successfully")
            except Exception as rollback_error:
                logger.error(f"[DB_SAVE] Error during rollback: {rollback_error}")
            raise
    
    def test_database_connection(self) -> bool:
        """
        Test database connection health.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            logger.debug("[DB_TEST] Testing database connection")
            # Simple query to test connection
            db.session.execute(text('SELECT 1'))
            db.session.commit()
            logger.info("[DB_TEST] Database connection is healthy")
            return True
        except Exception as e:
            logger.error(f"[DB_TEST] Database connection test failed: {e}")
            try:
                db.session.rollback()
            except:
                pass
            return False
    
    def get_error_statistics(self) -> Dict[str, int]:
        """
        Get database operation error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return self.error_stats.copy()
    
    def reset_error_statistics(self):
        """Reset error statistics counters."""
        self.error_stats = {key: 0 for key in self.error_stats}
        logger.info("[DB_SAVE] Error statistics reset")