# 📖 Bidirectional RNN for Sentiment Analysis (IMDB Dataset)

This project implements a **Bidirectional Recurrent Neural Network (Bi-RNN)** using **TensorFlow/Keras** to classify movie reviews from the [IMDB dataset](https://keras.io/api/datasets/imdb/).  
The model captures both **forward** and **backward context** in sequences, making it more effective for natural language processing tasks like sentiment classification.

---

## 🚀 Features
- Loads and preprocesses the **IMDB movie review dataset**  
- Implements a **Bidirectional LSTM** architecture with embeddings  
- Tracks **training & validation curves** (loss, accuracy)  
- Includes **evaluation plots**:
  - Confusion Matrix (heatmap)
  - ROC Curve + AUC
  - Precision–Recall Curve
  - Probability Distribution of predictions
- Visualizes the **sequence relationship in a Bi-RNN**  
- Supports saving the trained model to **Google Drive**

---

## 🧰 Tech Stack
- Python 3.x  
- TensorFlow / Keras  
- NumPy, Matplotlib, Seaborn  
- scikit-learn  
- Google Colab (GPU support recommended)

---

## ⚙️ Model Architecture
- **Embedding Layer**: 20k vocabulary, 128-dim embeddings  
- **Bidirectional LSTM**: 64 hidden units  
- **Dropout**: regularization to prevent overfitting  
- **Dense Layers**: 64 (ReLU) + 1 (Sigmoid) for binary classification  

---

## 📊 Example Results
- Test Accuracy: ~87% (varies by run)  
- ROC AUC: ~0.92  
- Training & validation loss curves show stable convergence  

Confusion Matrix Example:

|               | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| **True Neg**  | 11,800             | 1,200              |
| **True Pos**  | 1,000              | 12,000             |

---

## 📈 Visualizations

- Training Curves (loss & accuracy)
- Confusion Matrix
- ROC & Precision–Recall Curves
- Probability distribution of predicted sentiments
- Bi-RNN sequence flow diagram
