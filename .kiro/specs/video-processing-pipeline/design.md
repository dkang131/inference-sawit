# Design Document

## Overview

The video processing pipeline is a Flask-based web application that enables users to upload videos, perform YOLO object detection inference, view real-time processing results, and save detection counts to a PostgreSQL database. The system addresses the critical issue of premature video processing termination by implementing robust error handling, proper thread management, and comprehensive progress tracking.

## Architecture

The system follows a multi-threaded architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Flask Backend  │    │   Database      │
│                 │    │                 │    │   (PostgreSQL)  │
│  - File Upload  │◄──►│  - API Routes   │◄──►│  - Video Records│
│  - Model Select │    │  - Video Stream │    │  - Count Data   │
│  - Live Feed    │    │  - Status API   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Video Processor │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ Capture     │ │
                    │ │ Thread      │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │ Process     │ │
                    │ │ Thread      │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │ YOLO        │ │
                    │ │ Counter     │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

## Components and Interfaces

### 1. Web Application Layer (Flask)

**Purpose**: Handle HTTP requests, serve the user interface, and manage API endpoints.

**Key Components**:
- `dashboard/app.py`: Main Flask application with API routes
- Template rendering for the user interface
- File upload handling and validation
- Video streaming endpoint for live feed

**API Endpoints**:
- `POST /api/upload`: Handle video file uploads
- `POST /api/start_processing`: Initialize video processing with selected model
- `GET /api/processing_status`: Return current processing status and results
- `GET /api/video_feed`: Stream processed video frames
- `POST /api/stop_processing`: Manually stop processing and save results
- `GET /api/results`: Retrieve historical processing results
- `GET /api/output_video/<id>`: Download processed video files

### 2. Video Processing Engine

**Purpose**: Handle video frame capture, processing, and YOLO inference in a multi-threaded environment.

**Key Components**:

#### VideoProcess Class (`utils/video_processor.py`)
- **Frame Capture Thread**: Reads video frames and queues them for processing
- **Frame Processing Thread**: Applies YOLO inference and object counting
- **Progress Tracking**: Monitors processing progress and completion status
- **Error Recovery**: Handles video stream corruption and read failures

**Critical Design Decisions**:
- **Dual-threaded approach**: Separate capture and processing threads prevent blocking
- **Frame queue with size limit**: Prevents memory overflow for long videos
- **Graceful error handling**: Continue processing on individual frame failures
- **Progress monitoring**: Track frame position vs total frames for accurate progress
- **Video reinitialization**: Recover from stream corruption by reopening video file

#### Object Counter (`utils/counter.py`)
- **SilentObjectCounter**: Custom YOLO counter that doesn't display windows
- **Model management**: Dynamic model loading based on user selection
- **Region-based counting**: Configurable detection regions for accurate counting
- **Class-wise tracking**: Track different object types separately

### 3. Database Layer

**Purpose**: Persist video processing results and metadata.

**Schema** (`models.py`):
```python
class Video(db.Model):
    id: Primary key
    video_file_name: Original filename
    model_name: YOLO model used for processing
    total_count: Total objects detected
    *_count: Class-specific counts (empty, hb, unripe, ripe, overripe)
    *_ls_count: Long Stalk variants
    *_ab_count: Abnormal variants
    bunch_count: Fruit Bunch
    created_at: Processing timestamp
```

## Data Models

### Processing Result Structure
```python
{
    'frame': numpy.ndarray,           # Annotated video frame
    'in_count': int,                  # Total objects entering region
    'total_tracks': int,              # Active object tracks
    'classwise_counts': dict,         # Per-class detection counts
    'fps': float,                     # Current processing FPS
    'frame_count': int,               # Current frame number
    'progress': float,                # Processing progress percentage
    'is_complete': bool,              # Processing completion status
    'timestamp': float                # Result timestamp
}
```

### Video Processing State
```python
{
    'processing': bool,               # Processing active flag
    'frames_processed': int,          # Number of frames processed
    'total_frames': int,              # Total frames in video
    'consecutive_read_failures': int, # Error tracking counter
    'end_of_video': bool,            # End of video reached flag
    'processing_complete': bool       # Processing finished flag
}
```

## Error Handling

### 1. Video Stream Corruption
- **Detection**: Monitor consecutive read failures
- **Recovery**: Reinitialize video capture and seek to current position
- **Fallback**: Continue processing from last known good position

### 2. Frame Processing Errors
- **Isolation**: Catch exceptions per frame to prevent pipeline termination
- **Logging**: Record detailed error information for debugging
- **Continuation**: Skip problematic frames and continue with next frame

### 3. Model Loading Failures
- **Validation**: Check model file existence before processing
- **Fallback**: Use default model if specified model unavailable
- **User Feedback**: Return clear error messages for model issues

### 4. Database Connection Issues
- **Retry Logic**: Attempt database operations with timeout
- **Graceful Degradation**: Continue processing even if database save fails
- **User Notification**: Inform user of database issues while preserving results

### 5. Memory Management
- **Frame Queue Limits**: Prevent memory overflow with bounded queues
- **Resource Cleanup**: Properly release video capture and writer resources
- **Thread Management**: Ensure threads terminate gracefully on stop

## Testing Strategy

### 1. Unit Testing
- **Video Processing Components**: Test frame capture, processing, and error handling
- **Database Operations**: Verify CRUD operations and data integrity
- **API Endpoints**: Test request/response handling and error cases
- **Model Loading**: Validate model selection and fallback mechanisms

### 2. Integration Testing
- **End-to-End Processing**: Test complete video processing pipeline
- **Multi-threading**: Verify thread coordination and synchronization
- **Error Recovery**: Test system behavior under various failure conditions
- **Database Integration**: Verify data persistence and retrieval

### 3. Performance Testing
- **Long Video Processing**: Test with videos of various lengths and resolutions
- **Memory Usage**: Monitor memory consumption during processing
- **Processing Speed**: Measure FPS and processing efficiency
- **Concurrent Users**: Test multiple simultaneous processing sessions

### 4. User Acceptance Testing
- **File Upload**: Test various video formats and sizes
- **Model Selection**: Verify model switching functionality
- **Live Feed**: Test real-time video streaming
- **Results Retrieval**: Verify processed video download and database queries

## Implementation Considerations

### 1. Thread Safety
- Use thread-safe queues for frame passing
- Implement proper synchronization for shared state
- Ensure graceful thread termination on stop requests

### 2. Resource Management
- Implement proper cleanup for video capture and writer objects
- Monitor and limit memory usage for long video processing
- Handle file system operations with appropriate error checking

### 3. Scalability
- Design for single-user processing (current scope)
- Consider future multi-user support with process isolation
- Implement efficient video streaming for real-time feedback

### 4. User Experience
- Provide real-time progress updates during processing
- Enable manual stop functionality with partial result saving
- Offer clear error messages and recovery suggestions

### 5. Configuration Management
- Support dynamic model selection from available models
- Allow configurable processing parameters (region points, thresholds)
- Maintain backward compatibility with existing database schema