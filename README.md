# 🔊 Voice Authentication System

## 📌 Overview
This **Voice Authentication System** verifies a user's identity based on their voice using **MFCC features** and a **three-step verification process**. The system allows users to **upload or record voice samples**, modify the audio (lower frequency & change pitch), and compare them for authentication.

## 🚀 Features
- ✅ **Upload or Record Audio** (WAV/MP3 supported)
- ✅ **Extract MFCC Features** for voice recognition
- ✅ **Three-Step Verification**:
  - **Step 1:** Direct voice comparison (**≥70% match ✅**)
  - **Step 2:** Lower frequency & compare (**≥50% match ✅**)
  - **Step 3:** Change pitch & compare (**≥50% match ✅**)
- ✅ **Real-time Similarity Scores**
- ✅ **Streamlit Web Interface**
- ✅ **Easy Deployment** on **Streamlit Cloud**

---

## 📂 Project Structure
```
📂 VoiceAuthApp
│── 📜 voice_auth.py           # Voice Authentication Functions
│── 📜 requirements.txt        # Python Dependencies
│── 📜 README.md               # Project Documentation
│── 📂 audio_samples           # Sample Voice Files
```

---

## 🛠️ Installation Guide
### 1️⃣ **Set Up Python Environment**
Ensure **Python 3.9+** is installed. You can download it from [python.org](https://www.python.org/downloads/).

### 2️⃣ **Install Dependencies**
Run the following command:
```sh
pip install -r requirements.txt
```
🔹 If `pyaudio` gives errors, install it manually:
```sh
pip install pipwin
pipwin install pyaudio
```

### 3️⃣ **Run the Streamlit App**
Navigate to the project folder and run:
```sh
streamlit run app.py
```

---

## 🎤 How to Use
1️⃣ **Upload a Reference Audio** (your voice sample) 📂  
2️⃣ **Upload an Input Audio** (new voice sample to verify) 🎤  
3️⃣ Click **Authenticate Voice** 🔍  
4️⃣ **View the results** with similarity percentages ✅❌  

---

## 🎯 Future Enhancements
- 🔹 **Real-time Voice Recording** in Streamlit UI 🎙️  
- 🔹 **Database for Multi-User Authentication** 🛡️  
- 🔹 **Deploy as a Web App** using Streamlit Cloud 🌍  
- 🔹 **AI-based Spoof Detection** to prevent replay attacks 🛑  

---

## 🤝 Contributing
We welcome contributions to improve the Voice Authentication System! 🚀  
To contribute:
1. **Fork** the repository.
2. **Create a new branch** (`feature-branch`).
3. **Make your changes** and submit a **pull request**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🌟 Show Your Support
If you like this project, **give it a star ⭐ on GitHub**!

---

## 📬 Contact
For any queries, feel free to **open an issue** or reach out on GitHub!

