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

