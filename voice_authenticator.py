"""
Voice Authentication Module

This module implements a three-step voice authentication algorithm using
MFCC feature extraction and cosine similarity measurements.

Algorithm Steps:
1. Direct MFCC comparison
2. Lower frequency MFCC comparison  
3. Pitch-shifted MFCC comparison

Author: Research Team
Date: August 2025
"""

import numpy as np
import scipy.spatial.distance as dist
from audio_processor import AudioProcessor
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAuthenticator:
    """
    A comprehensive voice authentication system using multi-step verification.
    """
    
    def __init__(self, 
                 similarity_threshold_direct: float = 0.7,
                 similarity_threshold_low: float = 0.5,
                 similarity_threshold_pitch: float = 0.5,
                 n_mfcc: int = 13):
        """
        Initialize the Voice Authenticator.
        
        Parameters:
        -----------
        similarity_threshold_direct : float, default=0.7
            Threshold for direct comparison
        similarity_threshold_low : float, default=0.5
            Threshold for low frequency comparison
        similarity_threshold_pitch : float, default=0.5
            Threshold for pitch shift comparison
        n_mfcc : int, default=13
            Number of MFCC coefficients
        """
        self.thresholds = {
            'direct': similarity_threshold_direct,
            'low_freq': similarity_threshold_low,
            'pitch_shift': similarity_threshold_pitch
        }
        self.audio_processor = AudioProcessor(n_mfcc=n_mfcc)
        
    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        
        Parameters:
        -----------
        features1 : np.ndarray
            First feature vector
        features2 : np.ndarray
            Second feature vector
            
        Returns:
        --------
        float
            Cosine similarity score (0-1)
        """
        try:
            return 1 - dist.cosine(features1, features2)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def _step1_direct_comparison(self, ref_signal: np.ndarray, 
                                input_signal: np.ndarray, 
                                sample_rate: int) -> float:
        """
        Step 1: Direct MFCC feature comparison.
        
        Parameters:
        -----------
        ref_signal : np.ndarray
            Reference audio signal
        input_signal : np.ndarray
            Input audio signal to verify
        sample_rate : int
            Sample rate of audio signals
            
        Returns:
        --------
        float
            Similarity score for direct comparison
        """
        logger.info("Performing Step 1: Direct MFCC comparison")
        
        ref_features = self.audio_processor.extract_mfcc_features(ref_signal, sample_rate)
        input_features = self.audio_processor.extract_mfcc_features(input_signal, sample_rate)
        
        similarity = self._compute_similarity(ref_features, input_features)
        logger.info(f"Step 1 similarity: {similarity:.4f}")
        
        return similarity
    
    def _step2_low_frequency_comparison(self, ref_signal: np.ndarray, 
                                       input_signal: np.ndarray, 
                                       sample_rate: int) -> float:
        """
        Step 2: Low frequency MFCC comparison after pitch shifting.
        
        Parameters:
        -----------
        ref_signal : np.ndarray
            Reference audio signal
        input_signal : np.ndarray
            Input audio signal to verify
        sample_rate : int
            Sample rate of audio signals
            
        Returns:
        --------
        float
            Similarity score for low frequency comparison
        """
        logger.info("Performing Step 2: Low frequency comparison")
        
        # Apply lower frequency transformation
        ref_low = self.audio_processor.apply_frequency_shift(ref_signal, sample_rate, n_steps=-3)
        input_low = self.audio_processor.apply_frequency_shift(input_signal, sample_rate, n_steps=-3)
        
        # Extract MFCC features from transformed signals
        ref_features_low = self.audio_processor.extract_mfcc_features(ref_low, sample_rate)
        input_features_low = self.audio_processor.extract_mfcc_features(input_low, sample_rate)
        
        similarity = self._compute_similarity(ref_features_low, input_features_low)
        logger.info(f"Step 2 similarity: {similarity:.4f}")
        
        return similarity
    
    def _step3_pitch_shift_comparison(self, ref_signal: np.ndarray, 
                                     input_signal: np.ndarray, 
                                     sample_rate: int) -> float:
        """
        Step 3: Pitch-shifted MFCC comparison.
        
        Parameters:
        -----------
        ref_signal : np.ndarray
            Reference audio signal
        input_signal : np.ndarray
            Input audio signal to verify
        sample_rate : int
            Sample rate of audio signals
            
        Returns:
        --------
        float
            Similarity score for pitch shift comparison
        """
        logger.info("Performing Step 3: Pitch shift comparison")
        
        # Apply pitch shift transformation
        ref_pitch = self.audio_processor.apply_frequency_shift(ref_signal, sample_rate, n_steps=3)
        input_pitch = self.audio_processor.apply_frequency_shift(input_signal, sample_rate, n_steps=3)
        
        # Extract MFCC features from transformed signals
        ref_features_pitch = self.audio_processor.extract_mfcc_features(ref_pitch, sample_rate)
        input_features_pitch = self.audio_processor.extract_mfcc_features(input_pitch, sample_rate)
        
        similarity = self._compute_similarity(ref_features_pitch, input_features_pitch)
        logger.info(f"Step 3 similarity: {similarity:.4f}")
        
        return similarity
    
    def authenticate(self, reference_audio_bytes: bytes, 
                    input_audio_bytes: bytes) -> Dict[str, any]:
        """
        Perform complete three-step voice authentication.
        
        Parameters:
        -----------
        reference_audio_bytes : bytes
            Reference audio file content
        input_audio_bytes : bytes
            Input audio file content to verify
            
        Returns:
        --------
        Dict[str, any]
            Authentication results containing similarity scores and final decision
        """
        try:
            logger.info("Starting voice authentication process")
            
            # Load audio files
            ref_signal, ref_sr = self.audio_processor.load_audio_from_bytes(reference_audio_bytes)
            input_signal, input_sr = self.audio_processor.load_audio_from_bytes(input_audio_bytes)
            
            # Ensure same sample rate (use reference sample rate)
            if input_sr != ref_sr:
                logger.warning(f"Sample rate mismatch: ref={ref_sr}, input={input_sr}")
                # You might want to resample here if needed
            
            # Preprocess audio
            ref_signal = self.audio_processor.preprocess_audio(ref_signal)
            input_signal = self.audio_processor.preprocess_audio(input_signal)
            
            # Perform three-step authentication
            step1_similarity = self._step1_direct_comparison(ref_signal, input_signal, ref_sr)
            step2_similarity = self._step2_low_frequency_comparison(ref_signal, input_signal, ref_sr)
            step3_similarity = self._step3_pitch_shift_comparison(ref_signal, input_signal, ref_sr)
            
            # Make authentication decision
            is_authenticated = (
                step1_similarity >= self.thresholds['direct'] and
                step2_similarity >= self.thresholds['low_freq'] and
                step3_similarity >= self.thresholds['pitch_shift']
            )
            
            # Calculate overall confidence score
            confidence_score = np.mean([step1_similarity, step2_similarity, step3_similarity])
            
            results = {
                'authenticated': is_authenticated,
                'confidence_score': confidence_score,
                'step1_direct_similarity': step1_similarity,
                'step2_low_freq_similarity': step2_similarity,
                'step3_pitch_shift_similarity': step3_similarity,
                'thresholds_used': self.thresholds.copy(),
                'individual_results': {
                    'step1_passed': step1_similarity >= self.thresholds['direct'],
                    'step2_passed': step2_similarity >= self.thresholds['low_freq'],
                    'step3_passed': step3_similarity >= self.thresholds['pitch_shift']
                }
            }
            
            logger.info(f"Authentication completed. Result: {'SUCCESS' if is_authenticated else 'FAILED'}")
            logger.info(f"Confidence score: {confidence_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Authentication failed with error: {str(e)}")
            return {
                'authenticated': False,
                'confidence_score': 0.0,
                'error': str(e),
                'step1_direct_similarity': 0.0,
                'step2_low_freq_similarity': 0.0,
                'step3_pitch_shift_similarity': 0.0
            }