# Advanced Brain Tumor Classification Platform

An elegant, high-performance web application designed for classifying brain MRI scans to detect the presence of brain tumors. The system uses a customized, fine-tuned Deep Convolutional Neural Network based on the **VGG19** architecture and features a premium modern dashboard interface for medical imaging analysis.

---

## 🧠 Model Architecture & Methodology

The model classifies brain MRI scans into two distinct categories:
1.  **Class 0 (No Brain Tumor):** No indications of neoplastic tissues or structural abnormalities.
2.  **Class 1 (Yes Brain Tumor):** Abnormal tissue growths or structural patterns indicating the presence of a tumor.

### Architecture Details
*   **Base Network:** VGG19 (initialized with ImageNet weights, fully unfrozen for deep fine-tuning).
*   **Input Dimensions:** 240 x 240 pixels (RGB).
*   **Classification Head:**
    *   `Flatten` layer.
    *   `Dense` layer of 4,608 units (ReLU activation).
    *   `Dropout` layer (20% drop rate for regularization).
    *   `Dense` layer of 1,152 units (ReLU activation).
    *   `Dense` output layer of 2 units (Softmax activation).
*   **Total Parameters:** ~140,946,370 trainable parameters.

---

## 💻 Tech Stack

*   **Backend:** Flask (Python Web Server)
*   **Deep Learning:** TensorFlow / Keras
*   **Image Processing:** OpenCV, Pillow (PIL)
*   **Frontend Styling:** Bootstrap 5 (Premium Glassmorphism & Radial Glow Aesthetics)
*   **Icons:** FontAwesome 6

---

## 📁 Repository Structure

```text
├── app.py                      # Flask Application Backend
├── vgg_unfrozen.h5             # Link/File containing custom VGG19 model weights
├── templates/
│   └── index.html              # Beautiful Front-end Dashboard
├── uploads/                    # Temporary storage for uploaded scans (automatically ignored)
├── model_weights/              # Pretrained checkpoint weights
├── brain_tumor_dataset/        # Brain MRI dataset
├── .gitignore                  # Git ignore rules (configured for model weights & datasets)
└── README.md                   # Project documentation (this file)
```

---

## 🚀 How to Run the Web Application

### 1. Prerequisites
Ensure you have Python 3.11+ installed (Anaconda environment is highly recommended). Install the required packages using pip:
```bash
pip install tensorflow flask opencv-python pillow numpy
```

### 2. Run the App
Launch the Flask server in your terminal from the project's root directory:
```bash
python app.py
```

Once loaded, the terminal will print:
`Model loaded. Check http://127.0.0.1:5000/`

### 3. Access the Platform
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

Upload an MRI scan (drag-and-drop or select file) and hit **Run Neurological Diagnosis** to classify the scan instantly.

---

## 🛠️ Development & Environment Specs

*   **OS:** Windows 10/11
*   **Python:** 3.13.9 (Anaconda environment)
*   **Deep Learning Backend:** TensorFlow 2.x with oneDNN optimization enabled.
