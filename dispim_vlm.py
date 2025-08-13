"""
DiSPIM Vision Language Model Integration

Real-time VLM processing for DiSPIM microscopy including:
- Image stream processing with efficient VLM encoders
- Adaptive experiment control based on VLM analysis
- Scientific image understanding and decision making
- Integration with Bluesky plans for smart microscopy
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from queue import Queue, Empty
import json

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available - VLM features disabled")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available - GPT-4V features disabled")


class VLMProvider(Enum):
    """Available VLM providers"""
    HUGGINGFACE = "huggingface"
    OPENAI_GPT4V = "openai_gpt4v"
    CLAUDE_VISION = "claude_vision"
    LOCAL_LLAVA = "local_llava"


@dataclass
class VLMDecision:
    """VLM analysis decision for microscopy control"""
    # Analysis results
    description: str = ""
    confidence: float = 0.0
    detected_features: List[str] = None
    
    # Acquisition control
    continue_scan: bool = True
    next_step: float = 0.5  # μm
    change_exposure: Optional[float] = None  # ms
    
    # Focus and positioning
    focus_quality: float = 0.0
    suggest_autofocus: bool = False
    optimal_z_predicted: Optional[float] = None
    
    # Experiment adaptation
    increase_resolution: bool = False
    change_roi: Optional[tuple] = None
    switch_channel: Optional[str] = None
    
    # Quality metrics
    processing_time: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.detected_features is None:
            self.detected_features = []
        self.timestamp = time.time()


@dataclass
class VLMConfig:
    """Configuration for VLM processing"""
    provider: VLMProvider = VLMProvider.HUGGINGFACE
    model_name: str = "microsoft/git-base-coco"  # Default lightweight model
    
    # Processing parameters
    max_image_size: tuple = (512, 512)
    batch_size: int = 1
    processing_interval: float = 1.0  # seconds
    
    # Analysis prompts
    analysis_prompt: str = "Analyze this microscopy image. Describe cellular structures and image quality."
    decision_prompt: str = "Based on this microscopy image, should we continue scanning or adjust parameters?"
    
    # Decision thresholds
    min_confidence: float = 0.7
    focus_quality_threshold: float = 0.5
    feature_detection_threshold: float = 0.3
    
    # Performance settings
    use_gpu: bool = True
    max_processing_time: float = 5.0  # seconds timeout


class VLMProcessor:
    """Base class for VLM image processing"""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._model = None
        self._processor = None
        self._is_initialized = False
        
    def initialize(self):
        """Initialize VLM model and processor"""
        raise NotImplementedError
        
    def process_image(self, image: np.ndarray, prompt: str = None) -> Dict[str, Any]:
        """Process single image with VLM"""
        raise NotImplementedError
        
    def make_decision(self, image: np.ndarray, metadata: Dict = None) -> VLMDecision:
        """Make acquisition decision based on image analysis"""
        raise NotImplementedError


class HuggingFaceVLM(VLMProcessor):
    """HuggingFace Vision-Language Model processor"""
    
    def initialize(self):
        """Initialize HuggingFace VLM"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available")
        
        try:
            device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
            
            self._processor = AutoProcessor.from_pretrained(self.config.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(self.config.model_name)
            self._model.to(device)
            
            self._is_initialized = True
            self.logger.info(f"Initialized HuggingFace VLM: {self.config.model_name} on {device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace VLM: {e}")
    
    def process_image(self, image: np.ndarray, prompt: str = None) -> Dict[str, Any]:
        """Process image with HuggingFace VLM"""
        if not self._is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image / image.max() * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Resize if necessary
            if pil_image.size != self.config.max_image_size:
                pil_image = pil_image.resize(self.config.max_image_size)
            
            # Process with model
            if prompt is None:
                prompt = self.config.analysis_prompt
            
            inputs = self._processor(text=prompt, images=pil_image, return_tensors="pt")
            
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=100)
            
            generated_text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            processing_time = time.time() - start_time
            
            return {
                'description': generated_text.strip(),
                'processing_time': processing_time,
                'model': self.config.model_name,
                'prompt': prompt
            }
            
        except Exception as e:
            self.logger.error(f"VLM processing failed: {e}")
            return {
                'description': f"Processing failed: {e}",
                'processing_time': time.time() - start_time,
                'error': True
            }
    
    def make_decision(self, image: np.ndarray, metadata: Dict = None) -> VLMDecision:
        """Make acquisition decision using HuggingFace VLM"""
        # Analyze image
        analysis = self.process_image(image, self.config.analysis_prompt)
        
        # Simple heuristic-based decision making
        # In practice, this would be more sophisticated
        description = analysis.get('description', '').lower()
        
        decision = VLMDecision(
            description=analysis.get('description', ''),
            processing_time=analysis.get('processing_time', 0.0)
        )
        
        # Simple keyword-based analysis
        if 'blur' in description or 'out of focus' in description:
            decision.focus_quality = 0.2
            decision.suggest_autofocus = True
        elif 'sharp' in description or 'clear' in description:
            decision.focus_quality = 0.9
        else:
            decision.focus_quality = 0.5
        
        # Feature detection
        features = []
        if 'cell' in description:
            features.append('cells')
            decision.confidence = 0.8
        if 'structure' in description:
            features.append('structures')
            decision.confidence = max(decision.confidence, 0.7)
        
        decision.detected_features = features
        
        # Step size adaptation
        if decision.confidence > 0.8:
            decision.next_step = 0.3  # Smaller steps for interesting regions
        elif decision.confidence < 0.4:
            decision.next_step = 1.0   # Larger steps for empty regions
        
        return decision


class OpenAIVLM(VLMProcessor):
    """OpenAI GPT-4V processor for high-quality analysis"""
    
    def initialize(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai library not available")
        
        self._client = openai.OpenAI()
        self._is_initialized = True
        self.logger.info("Initialized OpenAI GPT-4V processor")
    
    def process_image(self, image: np.ndarray, prompt: str = None) -> Dict[str, Any]:
        """Process image with GPT-4V"""
        if not self._is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert to base64 for API
            import base64
            from io import BytesIO
            
            if image.dtype != np.uint8:
                image = (image / image.max() * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            if prompt is None:
                prompt = self.config.analysis_prompt
            
            response = self._client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }],
                max_tokens=200
            )
            
            processing_time = time.time() - start_time
            
            return {
                'description': response.choices[0].message.content,
                'processing_time': processing_time,
                'model': 'gpt-4-vision-preview',
                'prompt': prompt
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI VLM processing failed: {e}")
            return {
                'description': f"Processing failed: {e}",
                'processing_time': time.time() - start_time,
                'error': True
            }
    
    def make_decision(self, image: np.ndarray, metadata: Dict = None) -> VLMDecision:
        """Make acquisition decision using GPT-4V"""
        decision_prompt = f"""
        {self.config.decision_prompt}
        
        Please analyze this microscopy image and provide a JSON response with:
        - description: brief description of what you see
        - confidence: confidence score (0-1) for the analysis
        - focus_quality: focus quality score (0-1)
        - detected_features: list of detected cellular/biological features
        - continue_scan: whether to continue scanning (true/false)
        - next_step: suggested next step size in micrometers
        - suggest_autofocus: whether autofocus is needed (true/false)
        """
        
        analysis = self.process_image(image, decision_prompt)
        
        # Parse JSON response (with fallback)
        try:
            import re
            json_match = re.search(r'\{.*\}', analysis.get('description', ''), re.DOTALL)
            if json_match:
                decision_dict = json.loads(json_match.group())
                
                decision = VLMDecision(
                    description=decision_dict.get('description', ''),
                    confidence=float(decision_dict.get('confidence', 0.5)),
                    focus_quality=float(decision_dict.get('focus_quality', 0.5)),
                    detected_features=decision_dict.get('detected_features', []),
                    continue_scan=decision_dict.get('continue_scan', True),
                    next_step=float(decision_dict.get('next_step', 0.5)),
                    suggest_autofocus=decision_dict.get('suggest_autofocus', False),
                    processing_time=analysis.get('processing_time', 0.0)
                )
                
                return decision
        except:
            pass
        
        # Fallback to simple decision
        return VLMDecision(
            description=analysis.get('description', ''),
            processing_time=analysis.get('processing_time', 0.0)
        )


class DiSPIMVLMProcessor:
    """Main VLM processor for DiSPIM integration"""
    
    def __init__(self, config: VLMConfig = None):
        self.config = config or VLMConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize VLM processor
        if self.config.provider == VLMProvider.HUGGINGFACE:
            self.vlm = HuggingFaceVLM(self.config)
        elif self.config.provider == VLMProvider.OPENAI_GPT4V:
            self.vlm = OpenAIVLM(self.config)
        else:
            raise ValueError(f"Unsupported VLM provider: {self.config.provider}")
        
        # Processing queue and thread
        self._processing_queue = Queue(maxsize=10)
        self._result_queue = Queue()
        self._processing_thread = None
        self._stop_processing = threading.Event()
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'decisions_made': 0,
            'autofocus_suggested': 0
        }
    
    def start_processing(self):
        """Start background VLM processing thread"""
        if self._processing_thread and self._processing_thread.is_alive():
            return
        
        self._stop_processing.clear()
        self._processing_thread = threading.Thread(target=self._processing_worker)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        self.logger.info("Started VLM processing thread")
    
    def stop_processing(self):
        """Stop background processing thread"""
        self._stop_processing.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        self.logger.info("Stopped VLM processing thread")
    
    def _processing_worker(self):
        """Background worker for processing images"""
        self.vlm.initialize()
        
        while not self._stop_processing.is_set():
            try:
                # Get image from queue
                item = self._processing_queue.get(timeout=1.0)
                image, metadata, callback = item
                
                # Process with VLM
                decision = self.vlm.make_decision(image, metadata)
                
                # Update statistics
                self.stats['images_processed'] += 1
                self.stats['total_processing_time'] += decision.processing_time
                self.stats['decisions_made'] += 1
                if decision.suggest_autofocus:
                    self.stats['autofocus_suggested'] += 1
                
                # Store result
                self._result_queue.put(decision)
                
                # Call callback if provided
                if callback:
                    callback(decision)
                
                self._processing_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"VLM processing error: {e}")
    
    def process_image_async(self, image: np.ndarray, metadata: Dict = None, 
                           callback: Callable = None) -> bool:
        """Queue image for asynchronous VLM processing"""
        try:
            self._processing_queue.put_nowait((image, metadata, callback))
            return True
        except:
            self.logger.warning("VLM processing queue full, dropping image")
            return False
    
    def get_latest_decision(self, timeout: float = 0.1) -> Optional[VLMDecision]:
        """Get latest VLM decision if available"""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def create_bluesky_callback(self) -> Callable:
        """Create callback for Bluesky plan integration"""
        
        def vlm_callback(image_data: np.ndarray, z_position: float, 
                        z_index: int) -> Dict[str, Any]:
            """Process image and return decision for adaptive acquisition"""
            
            metadata = {
                'z_position': z_position,
                'z_index': z_index,
                'timestamp': time.time()
            }
            
            # Process synchronously for immediate decision
            decision = self.vlm.make_decision(image_data, metadata)
            
            # Convert to dictionary for Bluesky
            decision_dict = asdict(decision)
            
            self.logger.info(f"VLM decision at z={z_position}: "
                           f"confidence={decision.confidence:.2f}, "
                           f"continue={decision.continue_scan}")
            
            return decision_dict
        
        return vlm_callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['images_processed'] > 0:
            stats['avg_processing_time'] = (stats['total_processing_time'] / 
                                           stats['images_processed'])
        else:
            stats['avg_processing_time'] = 0.0
        
        return stats


# Example VLM configurations for different use cases
def create_fast_vlm_config() -> VLMConfig:
    """Create config for fast, lightweight VLM processing"""
    return VLMConfig(
        provider=VLMProvider.HUGGINGFACE,
        model_name="microsoft/git-base-coco",
        max_image_size=(256, 256),
        processing_interval=0.5,
        max_processing_time=2.0
    )


def create_high_quality_vlm_config() -> VLMConfig:
    """Create config for high-quality VLM analysis"""
    return VLMConfig(
        provider=VLMProvider.OPENAI_GPT4V,
        max_image_size=(512, 512),
        processing_interval=2.0,
        max_processing_time=10.0,
        analysis_prompt="""Analyze this light sheet microscopy image in detail. 
                          Identify cellular structures, tissue organization, 
                          and image quality metrics."""
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create VLM processor
    config = create_fast_vlm_config()
    vlm_processor = DiSPIMVLMProcessor(config)
    
    print("DiSPIM VLM processor initialized")
    print(f"Provider: {config.provider.value}")
    print(f"Model: {config.model_name}")
    
    # Start processing (would normally be done during acquisition)
    vlm_processor.start_processing()
    
    # Example callback for Bluesky
    vlm_callback = vlm_processor.create_bluesky_callback()
    print("VLM callback created for Bluesky integration")
    
    # Cleanup
    vlm_processor.stop_processing()