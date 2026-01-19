# Indian Sign Language (ISL) Detection System 🇮🇳

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

## 📖 Overview

This project is a real-time computer vision application designed to bridge the communication gap for the hearing and speech-impaired community. It utilizes a webcam to capture hand gestures and translates **Indian Sign Language (ISL)** into text and speech.

Unlike American Sign Language (ASL), which is predominantly single-handed, ISL frequently utilizes double-handed gestures. This system is optimized to track and classify these complex bimanual configurations.

## 🛠️ Technical Architecture

The system follows a standard computer vision pipeline:

1.  **Input Acquisition:** Captures video frames via OpenCV.
2.  **Feature Extraction:** Utilizes **Google MediaPipe** to extract 21 distinct (x, y, z) landmarks per hand.
3.  **Preprocessing:** Normalizes landmark coordinates relative to the wrist to ensure distance invariance.
4.  **Classification:** Feeds the normalized data into a **[INSERT MODEL TYPE, e.g., LSTM / Random Forest / K-Nearest Neighbors]** classifier to predict the gesture.
5.  **Output:** Renders the predicted character/word on the screen and converts it to audio using `pyttsx3`.

[Image of MediaPipe hand landmarks topology]

## 🤖 AI-Assisted Development ("Vibe Coding")

This project was built using an **AI-Assisted Development** workflow. 

* **Core Logic:** Developed via iterative prompting and code generation with **Google Gemini**.
* **Optimization:** AI was used to refactor the landmark processing pipeline and debug real-time latency issues.
* **Philosophy:** This repository demonstrates how modern Large Language Models (LLMs) can be leveraged to rapidly prototype accessible technology and solve complex logic problems.

## 🚀 Features

* **Real-time Detection:** Low-latency recognition suitable for live conversation.
* **Bimanual Support:** Detects both single-handed and double-handed ISL gestures.
* **Text-to-Speech:** Integrated audio output for seamless communication.
* **Robustness:** Includes basic handling for lighting variations and hand occlusion.

[Image of flowchart for real-time sign language detection]

## 📦 Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`), MediaPipe
* **Machine Learning:** [e.g., TensorFlow / Scikit-Learn]
* **Data Handling:** NumPy, Pandas
* **Audio:** pyttsx3 (Text-to-Speech)

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/isl-detection.git
cd isl-detection

# AI-Powered Indian Sign Language (ISL) Recognition & Smart Typer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-yellow)

A real-time computer vision application that translates Indian Sign Language (ISL) hand gestures into text. This project leverages **MediaPipe** for robust hand landmark extraction and a custom **Neural Network** trained on ISL gestures (A-Z, 0-9).

It features a **"Smart Typer"** interface with auto-complete, clipboard integration, and high-quality UI rendering, designed to act as an assistive communication tool.

---

## 🚀 Key Features

* **Real-Time Detection:** Instantly recognizes single and double-handed ISL gestures via webcam.
* **Robust Landmark Tracking:** Uses MediaPipe to extract 42 hand landmarks (21 per hand), making it resilient to lighting changes.
* **Smart Auto-Complete:** Suggests words based on the current character sequence (e.g., typing "GOO" suggests "GOOD").
* **Hands-Free Mode:** "Hold-to-type" functionality allows typing without touching the keyboard.
* **Clipboard Integration:** One-key shortcut to copy constructed sentences to the system clipboard.
* **High-Quality UI:** Anti-aliased text rendering using Pillow (PIL) for a polished look.

---

## 🛠️ Technical Architecture

### 1. Data Pipeline
Instead of using raw images (CNNs), this project uses a **Landmark-based approach** for higher efficiency and speed.
* **Input:** Live video frame from OpenCV.
* **Extraction:** MediaPipe Hands detects hands and extracts $(x, y)$ coordinates for 21 points per hand.
* **Normalization:** Data is normalized and sorted (Left Hand vs. Right Hand) to ensure consistent input vectors.
* **Vectorization:** The 42 points are flattened into a **(1, 84)** feature vector.

### 2. Model Structure (MLP)
The classifier is a Feed-Forward Neural Network (Multi-Layer Perceptron) built with TensorFlow/Keras:
* **Input Layer:** 84 neurons (42 landmarks $\times$ 2 coordinates).
* **Hidden Layers:**
    * Dense (128 neurons, ReLU activation) + Dropout (0.2)
    * Dense (64 neurons, ReLU activation) + Dropout (0.2)
* **Output Layer:** Softmax activation (Classes: A-Z, 0-9).

---

## 📂 Project Structure


ISL-Smart-Typer/
│
├── data/                 # (Not included in repo) Place your dataset here
│   ├── A/
│   ├── B/
│   └── ...
│
├── create_dataset.py     # Script to convert raw images -> CSV landmarks
├── train_model.py        # Script to train the Neural Network
├── inference.py          # Main application for real-time detection
├── isl_keypoints.csv     # Generated feature dataset (CSV)
├── isl_model.h5          # Pre-trained Model file
├── label_mapping.npy     # Saved label classes
├── arial.ttf             # (Optional) Font file for UI
├── requirements.txt      # Python dependencies
└── README.md             # Project Documentation
