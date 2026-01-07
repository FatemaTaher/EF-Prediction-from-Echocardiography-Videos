# EF-Prediction-from-Echocardiography-Videos
Deep learning system for predicting Left Ventricular Ejection Fraction (EF) from echocardiography videos using 3D CNN + LSTM, trained on the EchoNet-Dynamic dataset.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/FatemaTaher100/ef-predictor)

An end-to-end **deep learning system** for predicting **Left Ventricular Ejection Fraction (EF)** from echocardiography videos using a **3D CNN + LSTM** architecture.  
The project includes data preprocessing, model training, evaluation, and deployment as an interactive web application.

---

## 🚀 Live Demo
The model is deployed as an interactive **Gradio web application** on Hugging Face Spaces.

👉 **Try the demo here:**  
https://huggingface.co/spaces/FatemaTaher100/ef-predictor

Users can upload echocardiography videos and receive:
- Predicted EF value (%)
- EF severity level (Normal / Mildly Reduced / Severely Reduced)
- Visual analytics and sample frames
- Prediction history tracking

---

## 🫀 Problem Overview
**Ejection Fraction (EF)** is a critical clinical metric used to assess cardiac function.  
Traditional EF estimation is time-consuming and subject to inter-observer variability.

This project aims to **automate EF prediction** from echocardiography videos using deep learning techniques that capture both:
- **Spatial features** (cardiac structures)
- **Temporal dynamics** (heart motion over time)

---

## 🧠 Model Architecture
- **3D Convolutional Neural Network (CNN)** for spatial feature extraction  
- **LSTM** for temporal sequence modeling  
- **Regression head** for continuous EF prediction  

**Input:** 20 RGB frames (112 × 112)  
**Output:** Predicted EF value (%)

---

## 📊 Dataset
- **EchoNet-Dynamic Dataset**
- Provided by **Stanford AIMI**
- Contains **10,000+ echocardiography videos**
- Each video labeled with ground-truth EF values

🔗 Dataset link:  
https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a

> ⚠️ The dataset is not included in this repository and must be requested from Stanford AIMI.

---

## ⚙️ Data Processing Pipeline
- Video loading and validation
- Uniform frame sampling across each video
- Frame resizing and normalization
- Custom PyTorch `Dataset` and `DataLoader` implementation

---

## 🏋️ Training & Evaluation
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Learning Rate Scheduler:** ReduceLROnPlateau  

### Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🌐 Web Application Features
- Upload echocardiography videos (MP4, AVI, MOV, MKV)
- Color-coded EF severity visualization
- EF gauge charts
- Sample extracted frames
- Prediction history and CSV export

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- OpenCV  
- NumPy & Pandas  
- Matplotlib & Plotly  
- Gradio  

---

## Project Demo

### Dashboard Preview
![Dashboard](images/pic1.png)

### Sample Video Frames
![Sample Frames](images/pic2.png)

---

## 🗂️ Project Structure
EF-Prediction-from-Echocardiography-Videos/
│
├── checkpoints/
│   └── best_ef_model.pth        # Trained 3D CNN + LSTM model checkpoint
│
├── FileList.csv                 # Video metadata / dataset indexing
│
├── Training.ipynb               # Model training & experimentation notebook
│
├── Interface.ipynb              # Gradio-based web application for EF prediction
│
└── README.md                    # Project documentation

