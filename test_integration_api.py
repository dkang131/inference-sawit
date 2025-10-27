"""
Integration tests for API endpoints

Tests video upload, processing start/stop, status endpoints, processed video download,
and database operations according to requirements 1.1, 1.2, 1.3, 5.1, 5.2, 5.3, 7.2, 7.5
"""

import pytest
import os
import tempfile
import json
import time
import threading
from unittest.mock import patch, MagicMock, Mock, mock_open
import cv2
import numpy as np
import requests


class TestAPIIntegration:
    """Integration tests for API endpoints using direct HTTP requests"""
    
    BASE_URL = "http://localhost:4786"  # Default Flask development server
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a sample video file for testing"""
        # Create a temporary video file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, 'test_video.mp4')
        
        # Create a simple test video using OpenCV with longer duration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        # Write 30 frames (1.5 seconds at 20 fps) to pass minimum duration validation
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some content to make it a valid video
            cv2.putText(frame, f'Frame {i}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        
        yield video_path
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def mock_model_file(self):
        """Create a mock model file for testing"""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, 'test_model.pt')
        
        # Create a dummy model file
        with open(model_path, 'wb') as f:
            f.write(b'dummy model data')
        
        yield model_path
        
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rmdir(temp_dir)
    
    def test_api_endpoints_with_mocking(self):
        """Test API endpoints using mocking instead of actual HTTP requests"""
        # Import the Flask app and test directly
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test 1: Video upload endpoint with mocked validation
            with patch('dashboard.app.VideoValidator') as mock_validator_class:
                mock_validator = mock_validator_class.return_value
                mock_validator.validate_video_file.return_value = {
                    'is_valid': True,
                    'error_message': None,
                    'file_info': {
                        'format': 'mp4',
                        'size_mb': 1.5,
                        'duration': 2.0,
                        'resolution': (640, 480)
                    },
                    'validation_details': {},
                    'warnings': []
                }
                
                with patch('builtins.open', mock_open(read_data=b'fake video data')):
                    response = client.post('/api/upload', data={
                        'video': (tempfile.NamedTemporaryFile(suffix='.mp4'), 'test_video.mp4')
                    })
                
                # Should succeed with mocked validation
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'message' in data
                assert 'video_path' in data
    
    def test_processing_status_endpoint(self):
        """Test processing status endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test when no processor is active
            with patch('dashboard.app.video_processor', None):
                response = client.get('/api/processing_status')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] == 'idle'
                assert data['processing'] == False
                assert data['progress']['percentage'] == 0
    
    def test_start_processing_endpoint(self):
        """Test start processing endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test with missing parameters
            response = client.post('/api/start_processing', json={})
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'Missing video_path or model_name'
            
            # Test with non-existent file
            response = client.post('/api/start_processing', json={
                'video_path': '/nonexistent/path.mp4',
                'model_name': 'test_model.pt'
            })
            
            assert response.status_code == 404
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'Video file not found'
    
    def test_stop_processing_endpoint(self):
        """Test stop processing endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test when no processor is active
            with patch('dashboard.app.video_processor', None):
                response = client.post('/api/stop_processing')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['success'] == True
                assert data['message'] == 'No active processing to stop'
    
    def test_results_endpoint(self):
        """Test results endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test empty results - should return empty list even without database
            response = client.get('/api/results')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'results' in data
            # Results should be empty list when no data exists
            assert isinstance(data['results'], list)
    
    def test_output_video_endpoint(self):
        """Test output video endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test non-existent video by mocking the Video.query.get call
            with patch('dashboard.app.Video') as mock_video_class:
                mock_video_class.query.get.return_value = None
                
                response = client.get('/api/output_video/999')
                assert response.status_code == 404
                data = json.loads(response.data)
                assert 'error' in data
                assert data['error'] == 'Video record not found'
    
    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test with no processor
            with patch('dashboard.app.video_processor', None):
                response = client.get('/api/health_check')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] == 'healthy'
                assert data['system']['video_processor_active'] == False
    
    def test_performance_metrics_endpoint(self):
        """Test performance metrics endpoint"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test with no processor
            with patch('dashboard.app.video_processor', None):
                response = client.get('/api/performance_metrics')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] == 'idle'
                assert data['metrics']['processing_fps'] == 0
    
    def test_video_file_management_endpoints(self):
        """Test video file management endpoints"""
        from dashboard.app import app
        
        with app.test_client() as client:
            # Test remove video without path
            response = client.post('/api/remove_video', json={})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['error'] == 'Missing video_path'
            
            # Test cleanup old videos - mock the actual cleanup call
            with patch('dashboard.app.VideoFileManager') as mock_manager_class:
                mock_manager = mock_manager_class.return_value
                mock_manager.cleanup_old_videos.return_value = {
                    'files_removed': 3,
                    'files_failed': 0,
                    'bytes_freed': 1024000
                }
                
                response = client.post('/api/cleanup_old_videos', json={'max_age_hours': 24})
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'cleanup_stats' in data
                assert data['cleanup_stats']['files_removed'] == 3
            
            # Test upload directory info
            with patch('dashboard.app.VideoFileManager') as mock_manager_class:
                mock_manager = mock_manager_class.return_value
                mock_manager.get_upload_directory_info.return_value = {
                    'file_count': 5,
                    'total_size_mb': 150.5,
                    'files': []
                }
                
                response = client.get('/api/upload_directory_info')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['directory_info']['file_count'] == 5
    
    def test_database_operations(self):
        """Test database operations and result saving - Requirements 5.1, 5.2, 5.3, 5.4, 5.5"""
        from utils.database_manager import DatabaseManager
        from models import Video
        
        # Test database manager functionality with mocking
        with patch('utils.database_manager.db') as mock_db:
            mock_session = MagicMock()
            mock_db.session = mock_session
            
            # Create database manager
            db_manager = DatabaseManager()
            
            # Test saving video results
            processing_results = {
                'in_count': 100,
                'total_tracks': 50,
                'classwise_counts': {
                    'ripe': 60,
                    'unripe': 30,
                    'empty': 10
                },
                'fps': 25.0,
                'frame_count': 1000
            }
            
            processing_metadata = {
                'processing_fps': 25.0,
                'total_frames_processed': 1000,
                'processing_duration': 40.0,
                'completion_method': 'auto_complete'
            }
            
            # Mock successful database save
            mock_video = MagicMock()
            mock_video.id = 123
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            
            with patch('utils.database_manager.Video') as mock_video_class:
                mock_video_class.return_value = mock_video
                
                video_id = db_manager.save_video_results(
                    video_filename='test_video.mp4',
                    model_name='test_model.pt',
                    processing_results=processing_results,
                    processing_metadata=processing_metadata
                )
                
                # Verify database operations were called
                mock_video_class.assert_called_once()
                mock_session.add.assert_called_once()
                mock_session.commit.assert_called_once()
                assert video_id == 123
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a sample video file for testing"""
        # Create a temporary video file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, 'test_video.mp4')
        
        # Create a simple test video using OpenCV with longer duration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        # Write 30 frames (1.5 seconds at 20 fps) to pass minimum duration validation
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some content to make it a valid video
            cv2.putText(frame, f'Frame {i}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        
        yield video_path
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def mock_model_file(self):
        """Create a mock model file for testing"""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, 'test_model.pt')
        
        # Create a dummy model file
        with open(model_path, 'wb') as f:
            f.write(b'dummy model data')
        
        yield model_path
        
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rmdir(temp_dir)




if __name__ == '__main__':
    pytest.main([__file__, '-v'])