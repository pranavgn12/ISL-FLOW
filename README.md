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

```bash
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
