"""
Configuration Module for Voice Authentication Research System

This module contains configuration parameters and constants used throughout
the voice authentication system. Modify these values to adjust system behavior.

Author: Research Team
Date: August 2025
"""

from typing import Dict, Any
import os
from pathlib import Path


class AuthenticationConfig:
    """Configuration class for authentication parameters."""
    
    # Default similarity thresholds
    DEFAULT_DIRECT_THRESHOLD = 0.7
    DEFAULT_LOW_FREQ_THRESHOLD = 0.5
    DEFAULT_PITCH_SHIFT_THRESHOLD = 0.5
    
    # MFCC feature extraction parameters
    DEFAULT_N_MFCC = 13
    DEFAULT_N_FFT = 2048
    DEFAULT_HOP_LENGTH = 512
    
    # Audio processing parameters
    TARGET_SAMPLE_RATE = 22050
    PITCH_SHIFT_STEPS_DOWN = -3  # For low frequency analysis
    PITCH_SHIFT_STEPS_UP = 3     # For pitch shift analysis
    
    # File format support
    SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            'similarity_thresholds': {
                'direct': cls.DEFAULT_DIRECT_THRESHOLD,
                'low_freq': cls.DEFAULT_LOW_FREQ_THRESHOLD,
                'pitch_shift': cls.DEFAULT_PITCH_SHIFT_THRESHOLD
            },
            'mfcc_params': {
                'n_mfcc': cls.DEFAULT_N_MFCC,
                'n_fft': cls.DEFAULT_N_FFT,
                'hop_length': cls.DEFAULT_HOP_LENGTH
            },
            'audio_params': {
                'target_sample_rate': cls.TARGET_SAMPLE_RATE,
                'pitch_shift_down': cls.PITCH_SHIFT_STEPS_DOWN,
                'pitch_shift_up': cls.PITCH_SHIFT_STEPS_UP
            },
            'file_support': {
                'formats': cls.SUPPORTED_AUDIO_FORMATS
            }
        }
    
    @classmethod
    def load_from_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        import json
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Merge with defaults
        default_config = cls.get_default_config()
        default_config.update(config)
        
        return default_config
    
    @classmethod
    def save_to_file(cls, config: Dict[str, Any], config_path: str):
        """Save configuration to JSON file."""
        import json
        config_file = Path(config_path)
        
        # Create directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)


class ResearchConfig:
    """Configuration for research-specific parameters."""
    
    # Experiment settings
    EXPERIMENT_RESULTS_DIR = Path("experiments/results")
    EXPERIMENT_DATA_DIR = Path("experiments/data")
    
    # Performance evaluation
    CROSS_VALIDATION_FOLDS = 5
    TEST_TRAIN_SPLIT = 0.2
    
    # Robustness testing parameters
    NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3]  # SNR ratios
    PITCH_VARIATIONS = [-6, -3, -1, 1, 3, 6]     # Semitones
    SPEED_VARIATIONS = [0.8, 0.9, 1.1, 1.2]     # Speed multipliers
    
    # Statistical analysis
    CONFIDENCE_INTERVAL = 0.95
    SIGNIFICANCE_LEVEL = 0.05
    
    @classmethod
    def create_experiment_dirs(cls):
        """Create necessary directories for experiments."""
        cls.EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_experiment_config(cls) -> Dict[str, Any]:
        """Get experiment configuration dictionary."""
        return {
            'directories': {
                'results': str(cls.EXPERIMENT_RESULTS_DIR),
                'data': str(cls.EXPERIMENT_DATA_DIR)
            },
            'evaluation': {
                'cv_folds': cls.CROSS_VALIDATION_FOLDS,
                'test_split': cls.TEST_TRAIN_SPLIT
            },
            'robustness_testing': {
                'noise_levels': cls.NOISE_LEVELS,
                'pitch_variations': cls.PITCH_VARIATIONS,
                'speed_variations': cls.SPEED_VARIATIONS
            },
            'statistics': {
                'confidence_interval': cls.CONFIDENCE_INTERVAL,
                'significance_level': cls.SIGNIFICANCE_LEVEL
            }
        }


class StreamlitConfig:
    """Configuration for Streamlit web interface."""
    
    # Page configuration
    PAGE_TITLE = "Voice Authentication Research System"
    PAGE_ICON = "🔊"
    LAYOUT = "wide"
    
    # UI parameters
    MAX_FILE_SIZE_MB = 10
    ALLOWED_FILE_TYPES = ["wav", "mp3", "m4a", "flac"]
    
    # Chart configurations
    CHART_HEIGHT = 400
    CHART_COLOR_SCHEME = "viridis"
    
    # Performance settings
    CACHE_TTL = 3600  # 1 hour
    
    @classmethod
    def get_streamlit_config(cls) -> Dict[str, Any]:
        """Get Streamlit configuration dictionary."""
        return {
            'page': {
                'title': cls.PAGE_TITLE,
                'icon': cls.PAGE_ICON,
                'layout': cls.LAYOUT
            },
            'file_upload': {
                'max_size_mb': cls.MAX_FILE_SIZE_MB,
                'allowed_types': cls.ALLOWED_FILE_TYPES
            },
            'charts': {
                'height': cls.CHART_HEIGHT,
                'color_scheme': cls.CHART_COLOR_SCHEME
            },
            'performance': {
                'cache_ttl': cls.CACHE_TTL
            }
        }


# Environment-based configuration loading
def load_config() -> Dict[str, Any]:
    """Load configuration based on environment."""
    config_file = os.getenv('VOICE_AUTH_CONFIG', 'config.json')
    
    if os.path.exists(config_file):
        return AuthenticationConfig.load_from_file(config_file)
    else:
        return AuthenticationConfig.get_default_config()


# Global configuration instance
CONFIG = load_config()

# Export commonly used values
SIMILARITY_THRESHOLDS = CONFIG['similarity_thresholds']
MFCC_PARAMS = CONFIG['mfcc_params']
AUDIO_PARAMS = CONFIG['audio_params']

if __name__ == "__main__":
    # Example usage and configuration validation
    print("Voice Authentication System Configuration")
    print("=" * 50)
    
    # Display default configuration
    default_config = AuthenticationConfig.get_default_config()
    
    import json
    print("Default Configuration:")
    print(json.dumps(default_config, indent=2))
    
    # Create experiment directories
    ResearchConfig.create_experiment_dirs()
    print(f"\nExperiment directories created:")
    print(f"Results: {ResearchConfig.EXPERIMENT_RESULTS_DIR}")
    print(f"Data: {ResearchConfig.EXPERIMENT_DATA_DIR}")
    
    # Save example configuration file
    example_config_path = "example_config.json"
    AuthenticationConfig.save_to_file(default_config, example_config_path)
    print(f"\nExample configuration saved to: {example_config_path}")