"""
Audio Processing Module for Voice Authentication System

This module contains functions for audio feature extraction and audio modifications
used in the three-step voice authentication process.

Author: Research Team
Date: August 2025
"""

import librosa
import numpy as np
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    A class to handle audio processing operations for voice authentication.
    """
    
    def __init__(self, n_mfcc=13):
        """
        Initialize the AudioProcessor.
        
        Parameters:
        -----------
        n_mfcc : int, default=13
            Number of MFCC coefficients to extract
        """
        self.n_mfcc = n_mfcc
    
    def extract_mfcc_features(self, audio_signal, sample_rate):
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features from audio signal.
        
        Parameters:
        -----------
        audio_signal : np.ndarray
            The audio time series
        sample_rate : int
            Sample rate of the audio signal
            
        Returns:
        --------
        np.ndarray
            Mean MFCC features across time frames
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_signal, 
                sr=sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            # Return mean across time axis
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            raise ValueError(f"Error extracting MFCC features: {str(e)}")
    
    def apply_frequency_shift(self, audio_signal, sample_rate, n_steps=-3):
        """
        Apply frequency/pitch shift to audio signal.
        
        Parameters:
        -----------
        audio_signal : np.ndarray
            The audio time series
        sample_rate : int
            Sample rate of the audio signal
        n_steps : int, default=-3
            Number of semitones to shift (negative for lower pitch)
            
        Returns:
        --------
        np.ndarray
            Pitch-shifted audio signal
        """
        try:
            return librosa.effects.pitch_shift(
                audio_signal, 
                sr=sample_rate, 
                n_steps=n_steps
            )
        except Exception as e:
            raise ValueError(f"Error applying pitch shift: {str(e)}")
    
    def load_audio_from_bytes(self, audio_bytes):
        """
        Load audio from byte stream.
        
        Parameters:
        -----------
        audio_bytes : bytes
            Audio file content in bytes
            
        Returns:
        --------
        tuple
            (audio_signal, sample_rate)
        """
        try:
            audio_stream = BytesIO(audio_bytes)
            audio_signal, sample_rate = librosa.load(audio_stream, sr=None)
            return audio_signal, sample_rate
        except Exception as e:
            raise ValueError(f"Error loading audio: {str(e)}")
    
    def preprocess_audio(self, audio_signal, target_sr=22050):
        """
        Preprocess audio signal (normalize, resample if needed).
        
        Parameters:
        -----------
        audio_signal : np.ndarray
            The audio time series
        target_sr : int, default=22050
            Target sample rate
            
        Returns:
        --------
        np.ndarray
            Preprocessed audio signal
        """
        # Normalize audio
        if np.max(np.abs(audio_signal)) > 0:
            audio_signal = audio_signal / np.max(np.abs(audio_signal))
        
        return audio_signal