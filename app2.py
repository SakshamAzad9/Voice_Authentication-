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
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)  # ✅ Corrected Order

# 📌 Modify Audio - Change Pitch
def change_pitch(y, sr):
    """Changes the pitch of an audio signal."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=3)  # ✅ Corrected Order

# 📌 Three-Step Verification
def verify_audio(reference_audio, input_audio):
    """Performs voice authentication in three steps."""
    y_ref, sr = librosa.load(BytesIO(reference_audio))
    y_inp, _ = librosa.load(BytesIO(input_audio))

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

# 🎤 Streamlit UI
st.title("🔊 Voice Authentication System")
st.write("Upload your reference and test audio files to authenticate.")

# Upload Audio Files
reference_audio = st.file_uploader("Upload Reference Audio", type=["wav", "mp3"])
input_audio = st.file_uploader("Upload Input Audio", type=["wav", "mp3"])

# Perform Authentication
if st.button("Authenticate Voice") and reference_audio and input_audio:
    ref_bytes = reference_audio.read()
    inp_bytes = input_audio.read()

    # Run verification
    similarity, similarity_low, similarity_pitch = verify_audio(ref_bytes, inp_bytes)

    # Display Results
    st.write(f"**Step 1: Direct Match** - {similarity * 100:.2f}%")
    st.write(f"**Step 2: Lower Frequency Match** - {similarity_low * 100:.2f}%")
    st.write(f"**Step 3: Pitch Shift Match** - {similarity_pitch * 100:.2f}%")

    if similarity >= 0.7 and similarity_low >= 0.5 and similarity_pitch >= 0.5:
        st.success("✅ Authentication Successful!")
    else:
        st.error("❌ Authentication Failed!")
