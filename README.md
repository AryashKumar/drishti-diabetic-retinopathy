# 🩺 DRISHTI – Diabetic Retinopathy Detection using Hybrid Deep Learning

## 📌 Overview
DRISHTI is an AI-powered system designed to detect and classify **Diabetic Retinopathy (DR)** from retinal fundus images using a **hybrid deep learning model (CNN + Vision Transformer)**.

The system enables **early detection**, reduces dependency on specialists, and supports **scalable healthcare solutions**, especially in rural areas.

---

## 🚀 Features
- 🔍 Automated DR detection from retinal images  
- 🧠 Hybrid architecture (CNN + ViT)  
- 📊 Classification into 5 severity levels  
- ⚖️ Handles class imbalance (SMOTE, DCGAN, class weighting)  
- 📈 High accuracy (~91.4%)  
- 🌐 FastAPI-based deployment  
- 🧾 Explainable AI (Grad-CAM visualization)

---

## 🧠 Model Architecture


### 🔹 Components:
- **CNN Models**: ResNet, DenseNet, EfficientNet (local features)
- **Vision Transformer (ViT)**: Global feature extraction
- **Feature Fusion**: Attention-based combination
- **Classifier**: Softmax layer for DR stage prediction

---

## 📂 Dataset
- Source: Kaggle Diabetic Retinopathy Dataset  
- Total Images: ~35,000  
- Classes:
  - No DR  
  - Mild  
  - Moderate  
  - Severe  
  - Proliferative DR  

---

## ⚙️ Preprocessing
- Resizing (224×224)  
- CLAHE (Contrast enhancement)  
- Noise removal  
- Normalization  
- Data Augmentation (rotation, flipping, zooming)

---

## ⚖️ Handling Class Imbalance
- SMOTE (oversampling)  
- DCGAN (synthetic image generation)  
- Class weighting  

---

## 📊 Results

| Model | Accuracy |
|------|--------|
| ResNet50 | 87.6% |
| DenseNet121 | 88.9% |
| EfficientNetB3 | 90.1% |
| ViT | 91.0% |
| **DRISHTI (Hybrid)** | **91.4%** |

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🧪 Tech Stack
- Python  
- TensorFlow / PyTorch  
- OpenCV  
- NumPy, Pandas  
- Matplotlib  
- FastAPI  

---

## ⚡ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/drishti-dr-detection.git

# Navigate to project folder
cd drishti-dr-detection

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app:app --reload
