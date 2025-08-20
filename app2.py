import streamlit as st
import librosa
import numpy as np
import scipy.spatial.distance as dist
from io import BytesIO

# 📌 Extract MFCC Features
def extract_mfcc(y, sr):
    """Extracts MFCC features from an audio signal."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# 📌 Modify Audio - Lower Frequency
def lower_frequency(y, sr):
    """Lowers the frequency of an audio signal."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)

# 📌 Modify Audio - Change Pitch
def change_pitch(y, sr):
    """Changes the pitch of an audio signal."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=3)

# 📌 Three-Step Verification
def verify_audio(reference_audio, input_audio):
    """Performs voice authentication in three steps."""
    try:
        # Load audio files with error handling
        y_ref, sr_ref = librosa.load(BytesIO(reference_audio))
        y_inp, sr_inp = librosa.load(BytesIO(input_audio))
        
        # Ensure same sampling rate
        if sr_ref != sr_inp:
            y_inp = librosa.resample(y_inp, orig_sr=sr_inp, target_sr=sr_ref)
        sr = sr_ref
        
        # Step 1: Direct Comparison
        ref_features = extract_mfcc(y_ref, sr)
        input_features = extract_mfcc(y_inp, sr)
        similarity = 1 - dist.cosine(ref_features, input_features)
        
        # Step 2: Lower Frequency Comparison
        y_ref_low = lower_frequency(y_ref, sr)
        y_inp_low = lower_frequency(y_inp, sr)
        ref_features_low = extract_mfcc(y_ref_low, sr)
        inp_features_low = extract_mfcc(y_inp_low, sr)
        similarity_low = 1 - dist.cosine(ref_features_low, inp_features_low)
        
        # Step 3: Pitch Shift Comparison
        y_ref_pitch = change_pitch(y_ref, sr)
        y_inp_pitch = change_pitch(y_inp, sr)
        ref_features_pitch = extract_mfcc(y_ref_pitch, sr)
        inp_features_pitch = extract_mfcc(y_inp_pitch, sr)
        similarity_pitch = 1 - dist.cosine(ref_features_pitch, inp_features_pitch)
        
        return similarity, similarity_low, similarity_pitch
        
    except Exception as e:
        st.error(f"Error processing audio files: {str(e)}")
        return 0, 0, 0

# 🎤 Streamlit UI
def main():
    st.title("🔊 Voice Authentication System")
    st.write("Upload your reference and test audio files to authenticate.")
    st.write("---")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference Audio")
        reference_audio = st.file_uploader(
            "Upload Reference Audio", 
            type=["wav", "mp3", "flac", "m4a"],
            key="ref_audio"
        )
        if reference_audio:
            st.audio(reference_audio, format='audio/wav')
    
    with col2:
        st.subheader("Test Audio")
        input_audio = st.file_uploader(
            "Upload Test Audio", 
            type=["wav", "mp3", "flac", "m4a"],
            key="test_audio"
        )
        if input_audio:
            st.audio(input_audio, format='audio/wav')
    
    st.write("---")
    
    # Authentication section
    if st.button("🔍 Authenticate Voice", type="primary", use_container_width=True):
        if reference_audio and input_audio:
            with st.spinner("Processing audio files..."):
                # Read audio data
                ref_bytes = reference_audio.read()
                inp_bytes = input_audio.read()
                
                # Reset file pointers for potential re-use
                reference_audio.seek(0)
                input_audio.seek(0)
                
                # Run verification
                similarity, similarity_low, similarity_pitch = verify_audio(ref_bytes, inp_bytes)
                
                # Display Results
                st.subheader("Authentication Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Step 1: Direct Match",
                        value=f"{similarity * 100:.2f}%",
                        delta=f"Threshold: 70%"
                    )
                
                with col2:
                    st.metric(
                        label="Step 2: Lower Frequency",
                        value=f"{similarity_low * 100:.2f}%",
                        delta=f"Threshold: 50%"
                    )
                
                with col3:
                    st.metric(
                        label="Step 3: Pitch Shift",
                        value=f"{similarity_pitch * 100:.2f}%",
                        delta=f"Threshold: 50%"
                    )
                
                st.write("---")
                
                # Authentication Decision
                if similarity >= 0.7 and similarity_low >= 0.5 and similarity_pitch >= 0.5:
                    st.success("✅ Authentication Successful!")
                    st.balloons()
                else:
                    st.error("❌ Authentication Failed!")
                    
                    # Detailed feedback
                    st.write("**Failure Analysis:**")
                    if similarity < 0.7:
                        st.write("- Direct comparison below threshold")
                    if similarity_low < 0.5:
                        st.write("- Lower frequency comparison below threshold")
                    if similarity_pitch < 0.5:
                        st.write("- Pitch shift comparison below threshold")
                
        else:
            st.warning("⚠️ Please upload both reference and test audio files.")
    
    # Research Information
    with st.expander("📚 Research Methodology"):
        st.write("""
        **Three-Step Voice Authentication Process:**
        
        1. **Direct Comparison**: Extracts MFCC features from original audio signals
        2. **Lower Frequency Analysis**: Tests robustness against frequency modifications
        3. **Pitch Shift Analysis**: Tests robustness against pitch modifications
        
        **MFCC Features**: 13 Mel-Frequency Cepstral Coefficients are extracted and averaged
        **Similarity Measure**: Cosine similarity between feature vectors
        **Authentication Thresholds**: 70% for direct, 50% for modified versions
        """)

if __name__ == "__main__":
    main()