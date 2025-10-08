# Emoji Prediction using BiLSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

This repository contains an end-to-end implementation of an **Emoji Prediction** model using **Bidirectional LSTM (BiLSTM)** networks.  
It demonstrates data preprocessing, tokenization, oversampling for class balance, and model training using TensorFlow/Keras.

---

## Overview

Given a short text or tweet, the model predicts the most appropriate emoji label based on contextual and semantic cues.  
This project focuses on building a deep learning pipeline that integrates classical NLP techniques with modern sequence modeling.

---

## Key Features

- Text preprocessing (cleaning, tokenization, lemmatization)
- Data balancing with RandomOverSampler
- Tokenization and sequence padding for neural networks
- BiLSTM model architecture using TensorFlow/Keras
- Model evaluation and visualization (accuracy, confusion matrix)

---

## Model Architecture

```
Embedding (128)
↓
Bidirectional LSTM (80 units)
↓
Bidirectional LSTM (80 units)
↓
Global Max Pooling
↓
Dropout (0.5)
↓
Dense (64, ReLU)
↓
Dropout (0.5)
↓
Dense (Softmax output)
```

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK, SpaCy |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Data Balancing | imbalanced-learn |

---

## Training Details

- **Epochs:** 15  
- **Batch Size:** 64  
- **Validation Split:** 30%  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Evaluation Metric:** Accuracy  

Check GPU availability:
```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

---

## Results

Validation accuracy after training:
```python
val_acc = history.history['val_accuracy']
print("Validation Accuracy:", val_acc[-1])
```

(Optional) Generate a confusion matrix:
```python
from sklearn.metrics import confusion_matrix, classification_report
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt="d")
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/emoji_prediction.git
cd emoji_prediction
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install numpy pandas tensorflow nltk matplotlib seaborn scikit-learn
```

### 4. Run the Project
Ensure `data.csv` is in the correct directory:
```bash
python emoji_predictor.py
```

---

## Folder Structure

```
emoji_prediction/
│
├── src/
│   └── emoji_predictor.py
│
├── data.csv
├── emoji_mapping.txt
└── README.md
```

---

## Future Work

- Integrate pre-trained embeddings (GloVe, Word2Vec)
- Experiment with transformer-based models (BERT)
- Build an interactive web interface using Flask or Streamlit
- Extend to multilingual emoji prediction

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
