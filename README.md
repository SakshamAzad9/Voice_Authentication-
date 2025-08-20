# 🔊 Voice Authentication System

A **three-step voice authentication system** built with **Streamlit** and **Librosa**. This project verifies whether two audio samples belong to the same speaker by analyzing **MFCC features** and testing robustness against **frequency** and **pitch variations**.

---

## 🚀 Features

* **MFCC Feature Extraction**: Extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs).
* **Three-Step Authentication**:

  1. **Direct Comparison** – Compares MFCC features directly.
  2. **Lower Frequency Analysis** – Tests robustness against frequency shifts.
  3. **Pitch Shift Analysis** – Tests robustness against pitch changes.
* **Cosine Similarity Matching**: Computes similarity between reference and test audio.
* **Threshold-based Decision**:

  * Direct Match ≥ **70%**
  * Lower Frequency ≥ **50%**
  * Pitch Shift ≥ **50%**
* **Streamlit UI** with:

  * File uploaders for reference and test audio
  * Audio preview players
  * Authentication metrics and decision results
  * Interactive research methodology expander

---

## 🛠️ Tech Stack

* [Python 3.9+](https://www.python.org/)
* [Streamlit](https://streamlit.io/) – for UI
* [Librosa](https://librosa.org/) – for audio processing
* [NumPy](https://numpy.org/) – for numerical operations
* [SciPy](https://scipy.org/) – for similarity calculations

---

## 📦 Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/voice-authentication.git
   cd voice-authentication
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### requirements.txt

```txt
streamlit==1.28.1
librosa==0.10.1
numpy==1.24.3
scipy==1.11.1
soundfile==0.12.1
audioread==3.0.0
```

---

## ▶️ Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload a **Reference Audio** and a **Test Audio** file (supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`).

3. Click **Authenticate Voice** to run the three-step verification process.

4. View:

   * **Authentication metrics** (% similarity)
   * **Final decision** (Success/Failure)
   * **Failure Analysis** if thresholds are not met

---

## 📚 Research Methodology

* **MFCC Extraction**: 13 MFCCs are averaged over time to form feature vectors.
* **Cosine Similarity**: Used to measure closeness between feature vectors.
* **Three-Step Verification** ensures robustness:

  1. Direct comparison of MFCCs.
  2. Frequency-shifted analysis.
  3. Pitch-shifted analysis.
* **Thresholds**: Authentication requires all three conditions to pass.

---

## 📊 Example Results

| Step                     | Similarity (%) | Threshold (%) | Result |
| ------------------------ | -------------- | ------------- | ------ |
| Direct Comparison        | 85.32          | 70            | ✅ Pass |
| Lower Frequency Analysis | 65.21          | 50            | ✅ Pass |
| Pitch Shift Analysis     | 72.84          | 50            | ✅ Pass |

✅ **Authentication Successful** 🎉

---

## 📌 Future Improvements

* Support for **deep learning-based embeddings** (e.g., Speaker Verification with pretrained models).
* **Noise reduction & preprocessing** for real-world audio robustness.
* **Database integration** for storing and comparing multiple speaker profiles.

---

## 📝 License

This project is licensed under the **MIT License**. Feel free to use and modify it for your own projects.

