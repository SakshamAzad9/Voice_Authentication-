"""
Streamlit Web Interface for Voice Authentication System

A user-friendly web interface for the three-step voice authentication research system.
This interface allows users to upload reference and test audio files for authentication.

Author: Research Team
Date: August 2025

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from voice_authenticator import VoiceAuthenticator
import time
import logging

# Configure page
st.set_page_config(
    page_title="Voice Authentication System",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_similarity_chart(results):
    """Create a radar chart showing similarity scores for all three steps."""
    steps = ['Direct Match', 'Low Frequency', 'Pitch Shift']
    similarities = [
        results['step1_direct_similarity'] * 100,
        results['step2_low_freq_similarity'] * 100,
        results['step3_pitch_shift_similarity'] * 100
    ]
    thresholds = [
        results['thresholds_used']['direct'] * 100,
        results['thresholds_used']['low_freq'] * 100,
        results['thresholds_used']['pitch_shift'] * 100
    ]
    
    fig = go.Figure()
    
    # Add similarity scores
    fig.add_trace(go.Scatterpolar(
        r=similarities + [similarities[0]],  # Close the loop
        theta=steps + [steps[0]],
        fill='toself',
        name='Similarity Scores',
        line_color='rgba(0, 150, 136, 0.8)',
        fillcolor='rgba(0, 150, 136, 0.3)'
    ))
    
    # Add threshold lines
    fig.add_trace(go.Scatterpolar(
        r=thresholds + [thresholds[0]],
        theta=steps + [steps[0]],
        mode='lines',
        name='Thresholds',
        line=dict(color='rgba(255, 87, 51, 0.8)', dash='dash', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Authentication Step Results (%)",
        font=dict(size=12)
    )
    
    return fig

def create_confidence_gauge(confidence_score):
    """Create a gauge chart for overall confidence score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Confidence Score (%)"},
        delta = {'reference': 70},  # Reference threshold
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔊 Voice Authentication Research System")
    st.markdown("""
    This system implements a **three-step voice authentication algorithm** using MFCC feature extraction 
    and multiple audio transformations to improve robustness against voice spoofing attacks.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # Authentication thresholds
    st.sidebar.subheader("Authentication Thresholds")
    direct_threshold = st.sidebar.slider("Direct Match Threshold", 0.0, 1.0, 0.7, 0.05)
    low_freq_threshold = st.sidebar.slider("Low Frequency Threshold", 0.0, 1.0, 0.5, 0.05)
    pitch_threshold = st.sidebar.slider("Pitch Shift Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # MFCC parameters
    st.sidebar.subheader("Feature Extraction Parameters")
    n_mfcc = st.sidebar.slider("Number of MFCC Coefficients", 8, 20, 13)
    
    # Initialize authenticator with custom parameters
    authenticator = VoiceAuthenticator(
        similarity_threshold_direct=direct_threshold,
        similarity_threshold_low=low_freq_threshold,
        similarity_threshold_pitch=pitch_threshold,
        n_mfcc=n_mfcc
    )
    
    # Algorithm explanation
    with st.expander("📚 Algorithm Overview", expanded=False):
        st.markdown("""
        ### Three-Step Authentication Process:
        
        **Step 1: Direct MFCC Comparison**
        - Extract MFCC features from original audio signals
        - Compute cosine similarity between reference and input features
        
        **Step 2: Low Frequency Analysis**
        - Apply pitch shift (-3 semitones) to both signals
        - Extract MFCC features from transformed signals
        - Compare transformed features to detect frequency manipulation
        
        **Step 3: Pitch Shift Analysis**
        - Apply pitch shift (+3 semitones) to both signals
        - Extract MFCC features and compare
        - Provides additional robustness against voice synthesis attacks
        
        **Authentication Decision:** All three steps must pass their respective thresholds.
        """)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Reference Audio")
        reference_audio = st.file_uploader(
            "Choose reference audio file", 
            type=["wav", "mp3", "m4a", "flac"],
            help="Upload the reference voice sample for authentication"
        )
        
        if reference_audio:
            st.audio(reference_audio, format='audio/wav')
            st.success(f"✅ Reference audio loaded: {reference_audio.name}")
    
    with col2:
        st.subheader("📁 Upload Test Audio")
        input_audio = st.file_uploader(
            "Choose test audio file", 
            type=["wav", "mp3", "m4a", "flac"],
            help="Upload the audio sample to authenticate"
        )
        
        if input_audio:
            st.audio(input_audio, format='audio/wav')
            st.success(f"✅ Test audio loaded: {input_audio.name}")
    
    # Authentication button
    st.markdown("---")
    
    if st.button("🔍 Authenticate Voice", type="primary", use_container_width=True):
        if reference_audio and input_audio:
            
            with st.spinner("🔄 Processing audio files and performing authentication..."):
                # Read audio files
                ref_bytes = reference_audio.read()
                inp_bytes = input_audio.read()
                
                # Reset file pointers for potential re-reading
                reference_audio.seek(0)
                input_audio.seek(0)
                
                # Perform authentication
                start_time = time.time()
                results = authenticator.authenticate(ref_bytes, inp_bytes)
                processing_time = time.time() - start_time
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Authentication Results")
            
            # Overall result
            if results.get('authenticated', False):
                st.success("✅ **Authentication Successful!**")
                st.balloons()
            else:
                st.error("❌ **Authentication Failed!**")
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
            
            # Detailed results in columns
            if not results.get('error'):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    step1_result = "✅ PASS" if results['individual_results']['step1_passed'] else "❌ FAIL"
                    st.metric(
                        "Step 1: Direct Match", 
                        f"{results['step1_direct_similarity']*100:.2f}%",
                        delta=f"Threshold: {direct_threshold*100:.0f}%"
                    )
                    st.write(f"Result: {step1_result}")
                
                with col2:
                    step2_result = "✅ PASS" if results['individual_results']['step2_passed'] else "❌ FAIL"
                    st.metric(
                        "Step 2: Low Frequency", 
                        f"{results['step2_low_freq_similarity']*100:.2f}%",
                        delta=f"Threshold: {low_freq_threshold*100:.0f}%"
                    )
                    st.write(f"Result: {step2_result}")
                
                with col3:
                    step3_result = "✅ PASS" if results['individual_results']['step3_passed'] else "❌ FAIL"
                    st.metric(
                        "Step 3: Pitch Shift", 
                        f"{results['step3_pitch_shift_similarity']*100:.2f}%",
                        delta=f"Threshold: {pitch_threshold*100:.0f}%"
                    )
                    st.write(f"Result: {step3_result}")
                
                # Visualizations
                st.markdown("---")
                st.subheader("📈 Detailed Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Radar chart
                    radar_fig = create_similarity_chart(results)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with viz_col2:
                    # Confidence gauge
                    gauge_fig = create_confidence_gauge(results['confidence_score'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Performance metrics
                st.markdown("---")
                st.subheader("⚡ Performance Metrics")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                with perf_col2:
                    st.metric("Overall Confidence", f"{results['confidence_score']*100:.2f}%")
                
                with perf_col3:
                    passed_steps = sum(results['individual_results'].values())
                    st.metric("Steps Passed", f"{passed_steps}/3")
                
                with perf_col4:
                    st.metric("MFCC Coefficients", f"{n_mfcc}")
                
                # Export results
                if st.button("📥 Export Results as JSON"):
                    import json
                    st.download_button(
                        label="Download Results",
                        data=json.dumps(results, indent=2),
                        file_name=f"voice_auth_results_{int(time.time())}.json",
                        mime="application/json"
                    )
        else:
            st.warning("⚠️ Please upload both reference and test audio files.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Voice Authentication Research System | Built with Streamlit | August 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()