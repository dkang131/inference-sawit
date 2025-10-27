from ultralytics.solutions.object_counter import ObjectCounter  # pyright: ignore[reportMissingImports]
from ultralytics.solutions.solutions import SolutionResults, SolutionAnnotator  # pyright: ignore[reportMissingImports]
from ultralytics.utils.plotting import colors  # pyright: ignore[reportMissingImports]
import os
import logging
import sys
import traceback

# Configure logger with handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class SilentObjectCounter(ObjectCounter):
    """
    An ObjectCounter that performs all tracking and counting logic but does
    not display the counts or the final annotated image on the screen.
    It returns the annotated image array for further processing if needed.
    """

    def process(self, im0) -> SolutionResults:
        """
        This is an overridden version of the process method with CUDA error handling.

        It performs all the same logic as the original but skips the final
        display steps, preventing any windows from being created.
        """
        try:
            if not self.region_initialized:
                self.initialize_region()
                self.region_initialized = True

            self.extract_tracks(im0)
            
            # Fixed: Removed 'self.' from annotator initialization
            self.annotator = SolutionAnnotator(im0, line_width=self.line_width)

            self.annotator.draw_region(
                reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
            )

            for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
                self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
                self.store_tracking_history(track_id, box)

                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)

            plot_im = self.annotator.result()

            # Ensure we're creating the result correctly
            result = SolutionResults(
                plot_im=plot_im,
                in_count=getattr(self, 'in_count', 0),
                out_count=getattr(self, 'out_count', 0),
                classwise_count=dict(getattr(self, 'classwise_count', {})),
                total_tracks=len(getattr(self, 'track_ids', [])),
            )
            
            return result
            
        except (RuntimeError, Exception) as e:
            if 'CUDA' in str(e) or 'no kernel image' in str(e):
                logger.error(f"[PROCESS] CUDA error during processing: {e}")
                logger.info("[PROCESS] Attempting to move model to CPU")
                
                try:
                    # Force model to CPU
                    if hasattr(self, 'model') and hasattr(self.model, 'to'):
                        self.model.to('cpu')
                        logger.info("[PROCESS] Model moved to CPU successfully")
                        
                        # Retry processing
                        return self.process(im0)
                    else:
                        logger.error("[PROCESS] Cannot move model to CPU - model not accessible")
                        raise
                        
                except Exception as retry_error:
                    logger.error(f"[PROCESS] Failed to retry on CPU: {retry_error}")
                    raise
            else:
                logger.error(f"[PROCESS] Non-CUDA error during processing: {e}")
                raise


def validate_model_file(model_path):
    """
    Validate model file existence and basic properties
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        dict: Validation result with success status and details
    """
    try:
        logger.info(f"[MODEL_VALIDATION] Validating model file: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"[MODEL_VALIDATION] Model file does not exist: {model_path}")
            return {
                'is_valid': False,
                'error': 'Model file does not exist',
                'details': {'path': model_path, 'exists': False}
            }
        
        # Check if it's a file (not directory)
        if not os.path.isfile(model_path):
            logger.error(f"[MODEL_VALIDATION] Path is not a file: {model_path}")
            return {
                'is_valid': False,
                'error': 'Path is not a file',
                'details': {'path': model_path, 'is_file': False}
            }
        
        # Check file extension
        _, ext = os.path.splitext(model_path)
        valid_extensions = ['.pt', '.pth', '.onnx']
        if ext.lower() not in valid_extensions:
            logger.warning(f"[MODEL_VALIDATION] Unusual model file extension: {ext}")
            # Don't fail validation for unusual extensions, just warn
        
        # Check file size (should be > 0)
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            logger.error(f"[MODEL_VALIDATION] Model file is empty: {model_path}")
            return {
                'is_valid': False,
                'error': 'Model file is empty',
                'details': {'path': model_path, 'size_bytes': file_size}
            }
        
        # Check file permissions (readable)
        if not os.access(model_path, os.R_OK):
            logger.error(f"[MODEL_VALIDATION] Model file is not readable: {model_path}")
            return {
                'is_valid': False,
                'error': 'Model file is not readable',
                'details': {'path': model_path, 'readable': False}
            }
        
        logger.info(f"[MODEL_VALIDATION] Model file validation successful: {model_path} ({file_size} bytes)")
        return {
            'is_valid': True,
            'details': {
                'path': model_path,
                'size_bytes': file_size,
                'extension': ext,
                'readable': True
            }
        }
        
    except Exception as e:
        logger.error(f"[MODEL_VALIDATION] Error validating model file {model_path}: {e}")
        return {
            'is_valid': False,
            'error': f'Validation error: {str(e)}',
            'details': {'path': model_path, 'exception': str(e)}
        }


def get_default_model_path():
    """
    Get the path to the default model with fallback logic
    
    Returns:
        str or None: Path to default model, or None if no models available
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'model')
    
    if not os.path.exists(model_path):
        logger.error(f"[DEFAULT_MODEL] Model directory does not exist: {model_path}")
        return None
    
    # Try default model first
    default_model = os.path.join(model_path, 'model-latest.pt')
    if os.path.exists(default_model):
        logger.info(f"[DEFAULT_MODEL] Found default model: {default_model}")
        return default_model
    
    # Fall back to first available model
    try:
        models = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.pth', '.onnx'))]
        if models:
            # Sort models to ensure consistent selection
            models.sort()
            fallback_model = os.path.join(model_path, models[0])
            logger.info(f"[DEFAULT_MODEL] Using first available model as default: {fallback_model}")
            return fallback_model
    except Exception as e:
        logger.error(f"[DEFAULT_MODEL] Error scanning model directory: {e}")
    
    logger.error("[DEFAULT_MODEL] No models found in model directory")
    return None


def load_model_with_validation(model_name=None):
    """
    Load and validate model with comprehensive error handling and fallback logic
    
    Args:
        model_name (str, optional): Name of the model file to use
        
    Returns:
        dict: Result containing model path and validation details
    """
    logger.info(f"[MODEL_LOADING] Starting model loading with model_name: {model_name}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'model')
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        logger.error(f"[MODEL_LOADING] Model directory does not exist: {model_dir}")
        return {
            'success': False,
            'error': 'Model directory does not exist',
            'model_path': None,
            'model_name': None,
            'fallback_used': False
        }
    
    selected_model_path = None
    selected_model_name = None
    fallback_used = False
    
    # Try to use specified model first
    if model_name and model_name != 'No models available':
        specified_model_path = os.path.join(model_dir, model_name)
        logger.info(f"[MODEL_LOADING] Attempting to load specified model: {specified_model_path}")
        
        validation_result = validate_model_file(specified_model_path)
        if validation_result['is_valid']:
            selected_model_path = specified_model_path
            selected_model_name = model_name
            logger.info(f"[MODEL_LOADING] Successfully validated specified model: {model_name}")
        else:
            logger.warning(f"[MODEL_LOADING] Specified model validation failed: {validation_result['error']}")
            # Continue to fallback logic
    
    # Use fallback logic if specified model failed or wasn't provided
    if not selected_model_path:
        logger.info("[MODEL_LOADING] Using fallback model selection")
        fallback_model_path = get_default_model_path()
        
        if fallback_model_path:
            validation_result = validate_model_file(fallback_model_path)
            if validation_result['is_valid']:
                selected_model_path = fallback_model_path
                selected_model_name = os.path.basename(fallback_model_path)
                fallback_used = True
                logger.info(f"[MODEL_LOADING] Successfully validated fallback model: {selected_model_name}")
            else:
                logger.error(f"[MODEL_LOADING] Fallback model validation failed: {validation_result['error']}")
        else:
            logger.error("[MODEL_LOADING] No fallback model available")
    
    # Final validation
    if not selected_model_path:
        logger.error("[MODEL_LOADING] No valid model could be loaded")
        return {
            'success': False,
            'error': 'No valid model available',
            'model_path': None,
            'model_name': None,
            'fallback_used': fallback_used
        }
    
    logger.info(f"[MODEL_LOADING] Model loading successful - Path: {selected_model_path}, "
               f"Name: {selected_model_name}, Fallback used: {fallback_used}")
    
    return {
        'success': True,
        'model_path': selected_model_path,
        'model_name': selected_model_name,
        'fallback_used': fallback_used,
        'validation_details': validation_result['details'] if 'validation_result' in locals() else None
    }


def object_count(model_name=None):
    """
    Initialize object counter with enhanced model loading and validation
    
    Args:
        model_name: Name of the model file to use (e.g., 'model-latest.pt')
        
    Returns:
        SilentObjectCounter: Initialized counter object
        
    Raises:
        FileNotFoundError: If no valid model can be loaded
        RuntimeError: If model loading fails
    """
    logger.info(f"[OBJECT_COUNT] Initializing object counter with model: {model_name}")
    
    region_points = [(1602, 0), (1611, 1078)]
    
    # Load and validate model with enhanced error handling
    model_result = load_model_with_validation(model_name)
    
    if not model_result['success']:
        error_msg = f"Failed to load model: {model_result['error']}"
        logger.error(f"[OBJECT_COUNT] {error_msg}")
        raise FileNotFoundError(error_msg)
    
    model_path = model_result['model_path']
    actual_model_name = model_result['model_name']
    fallback_used = model_result['fallback_used']
    
    if fallback_used:
        logger.warning(f"[OBJECT_COUNT] Using fallback model '{actual_model_name}' instead of requested '{model_name}'")
    
    # Initialize counter with validated model and CUDA error handling
    try:
        logger.info(f"[OBJECT_COUNT] Initializing counter with region: {region_points} and model: {model_path}")
        
        # Force CPU-only mode due to RTX 5060 TI compatibility issues
        import torch
        import os
        
        # Check if CPU-only mode is forced via environment variables
        cuda_disabled = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        
        if cuda_disabled:
            logger.info("[OBJECT_COUNT] CUDA disabled via environment variable, using CPU")
            device = 'cpu'
            
            # Additional torch settings to force CPU
            torch.cuda.is_available = lambda: False
            logger.info("[OBJECT_COUNT] Overrode torch.cuda.is_available() to return False")
        else:
            cuda_available = torch.cuda.is_available()
            logger.info(f"[OBJECT_COUNT] CUDA available: {cuda_available}")
            
            if cuda_available:
                try:
                    # Test CUDA functionality
                    test_tensor = torch.randn(1, 3, 640, 640).cuda()
                    logger.info("[OBJECT_COUNT] CUDA test successful")
                    device = 'cuda'
                except Exception as cuda_error:
                    logger.warning(f"[OBJECT_COUNT] CUDA test failed: {cuda_error}")
                    logger.info("[OBJECT_COUNT] Falling back to CPU")
                    device = 'cpu'
            else:
                logger.info("[OBJECT_COUNT] CUDA not available, using CPU")
                device = 'cpu'
        
        # Initialize counter with device specification
        try:
            # Create counter with model path (let it handle device internally)
            counter = SilentObjectCounter(
                show=False,
                region=region_points,
                model=model_path,
                show_conf=True,
                show_labels=True, 
                show_in=True,
                show_out=False,
                verbose=True,
            )
            
            # Explicitly move model to the correct device using the proper method
            if hasattr(counter, 'model') and hasattr(counter.model, 'to'):
                counter.model.to(device)
                logger.info(f"[OBJECT_COUNT] Moved model to device: {device}")
            elif hasattr(counter, 'model'):
                logger.warning(f"[OBJECT_COUNT] Model doesn't have 'to' method, device may not be set correctly")
            
            # Force model to use the correct device
            if hasattr(counter, 'model') and hasattr(counter.model, 'to'):
                counter.model.to(device)
                logger.info(f"[OBJECT_COUNT] Model moved to device: {device}")
            else:
                logger.warning(f"[OBJECT_COUNT] Cannot move model to device {device} - model doesn't support 'to' method")
            
        except (RuntimeError, torch.cuda.CudaError) as cuda_error:
            if 'CUDA' in str(cuda_error) or 'no kernel image' in str(cuda_error):
                logger.warning(f"[OBJECT_COUNT] CUDA error during initialization: {cuda_error}")
                logger.info("[OBJECT_COUNT] Retrying with CPU-only mode")
                
                # Force CPU mode by setting environment variable
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # Reinitialize with CPU
                counter = SilentObjectCounter(
                    show=False,
                    region=region_points,
                    model=model_path,
                    show_conf=True,
                    show_labels=True, 
                    show_in=True,
                    show_out=False,
                    verbose=True,
                )
                
                # Ensure model is on CPU
                if hasattr(counter, 'model') and hasattr(counter.model, 'to'):
                    counter.model.to('cpu')
                    logger.info("[OBJECT_COUNT] Model forced to CPU due to CUDA compatibility issues")
                else:
                    logger.warning("[OBJECT_COUNT] Cannot force model to CPU - model doesn't support 'to' method")
                
                device = 'cpu'
            else:
                raise
        
        # Store model information in counter for later reference
        counter.model_path = model_path
        counter.model_name = actual_model_name
        counter.fallback_used = fallback_used
        counter.device = device
        
        logger.info(f"[OBJECT_COUNT] Counter initialized successfully with model: {actual_model_name} on device: {device}")
        return counter
        
    except Exception as e:
        error_msg = f"Failed to initialize counter with model '{actual_model_name}': {str(e)}"
        logger.error(f"[OBJECT_COUNT] {error_msg}")
        logger.error(f"[OBJECT_COUNT] Model path: {model_path}")
        traceback.print_exc()
        raise RuntimeError(error_msg) from e

