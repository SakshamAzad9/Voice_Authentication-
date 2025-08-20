"""
Main Demo Script for Voice Authentication Research System

This script demonstrates the voice authentication system functionality
without the Streamlit interface. Useful for research and testing purposes.

Author: Research Team
Date: August 2025

Usage:
    python main.py --reference ref_audio.wav --input test_audio.wav
"""

import argparse
import json
import logging
from pathlib import Path
from voice_authenticator import VoiceAuthenticator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_audio_file(file_path: str) -> bytes:
    """Load audio file and return as bytes."""
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise


def print_results(results: dict):
    """Pretty print authentication results."""
    print("\n" + "="*60)
    print("🔊 VOICE AUTHENTICATION RESULTS")
    print("="*60)
    
    # Overall result
    auth_status = "✅ SUCCESS" if results.get('authenticated', False) else "❌ FAILED"
    print(f"\nAuthentication Status: {auth_status}")
    print(f"Overall Confidence: {results.get('confidence_score', 0)*100:.2f}%")
    
    if results.get('error'):
        print(f"Error: {results['error']}")
        return
    
    print("\n" + "-"*40)
    print("STEP-BY-STEP RESULTS:")
    print("-"*40)
    
    # Step 1
    step1_status = "✅ PASS" if results['individual_results']['step1_passed'] else "❌ FAIL"
    print(f"Step 1 - Direct Match: {results['step1_direct_similarity']*100:.2f}% {step1_status}")
    print(f"  Threshold: {results['thresholds_used']['direct']*100:.0f}%")
    
    # Step 2
    step2_status = "✅ PASS" if results['individual_results']['step2_passed'] else "❌ FAIL"
    print(f"Step 2 - Low Frequency: {results['step2_low_freq_similarity']*100:.2f}% {step2_status}")
    print(f"  Threshold: {results['thresholds_used']['low_freq']*100:.0f}%")
    
    # Step 3
    step3_status = "✅ PASS" if results['individual_results']['step3_passed'] else "❌ FAIL"
    print(f"Step 3 - Pitch Shift: {results['step3_pitch_shift_similarity']*100:.2f}% {step3_status}")
    print(f"  Threshold: {results['thresholds_used']['pitch_shift']*100:.0f}%")
    
    # Summary
    passed_steps = sum(results['individual_results'].values())
    print(f"\nSummary: {passed_steps}/3 steps passed")
    print("="*60)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Voice Authentication Research System Demo"
    )
    parser.add_argument(
        "--reference", "-r",
        required=True,
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input audio file to authenticate"
    )
    parser.add_argument(
        "--direct-threshold", "-dt",
        type=float,
        default=0.7,
        help="Direct comparison threshold (default: 0.7)"
    )
    parser.add_argument(
        "--low-freq-threshold", "-lt",
        type=float,
        default=0.5,
        help="Low frequency comparison threshold (default: 0.5)"
    )
    parser.add_argument(
        "--pitch-threshold", "-pt",
        type=float,
        default=0.5,
        help="Pitch shift comparison threshold (default: 0.5)"
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients (default: 13)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path for results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate file paths
    ref_path = Path(args.reference)
    input_path = Path(args.input)
    
    if not ref_path.exists():
        logger.error(f"Reference audio file not found: {ref_path}")
        return 1
    
    if not input_path.exists():
        logger.error(f"Input audio file not found: {input_path}")
        return 1
    
    print("🎤 Voice Authentication Research System")
    print(f"Reference: {ref_path}")
    print(f"Input: {input_path}")
    
    try:
        # Initialize authenticator
        authenticator = VoiceAuthenticator(
            similarity_threshold_direct=args.direct_threshold,
            similarity_threshold_low=args.low_freq_threshold,
            similarity_threshold_pitch=args.pitch_threshold,
            n_mfcc=args.n_mfcc
        )
        
        # Load audio files
        logger.info("Loading audio files...")
        ref_bytes = load_audio_file(str(ref_path))
        input_bytes = load_audio_file(str(input_path))
        
        # Perform authentication
        logger.info("Starting authentication process...")
        start_time = time.time()
        results = authenticator.authenticate(ref_bytes, input_bytes)
        processing_time = time.time() - start_time
        
        # Add processing time to results
        results['processing_time_seconds'] = processing_time
        results['reference_file'] = str(ref_path)
        results['input_file'] = str(input_path)
        
        # Display results
        print_results(results)
        print(f"\nProcessing Time: {processing_time:.2f} seconds")
        
        # Save results to JSON if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
        
        # Return appropriate exit code
        return 0 if results.get('authenticated', False) else 1
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
    