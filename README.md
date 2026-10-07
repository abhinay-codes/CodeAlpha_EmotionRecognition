# Emotion Recognition from Speech

## Overview
This project builds a machine learning system to recognize human emotions from speech audio. It is a part of CodeAlpha Internship Task 2.

## Objective
The goal is to classify emotions from raw audio files by extracting MFCC (Mel-frequency cepstral coefficients) features and passing them through a deep learning classifier (Convolutional Neural Network - Conv1D) to predict emotions.

## Dataset
We use the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.
- Source: Zenodo (Audio_Speech_Actors_01-24.zip)
- Emotion Classes: Neutral (01), Calm (02), Happy (03), Sad (04), Angry (05), Fearful (06), Disgust (07), Surprised (08).
- Dataset limitations: The dataset consists of acted emotions in a clean recording environment, which may limit generalization to spontaneous real-world speech.

## Technologies
- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Methodology
```text
Audio
 ↓
Preprocessing (Loading with Librosa)
 ↓
MFCC Extraction (40 coefficients)
 ↓
Feature Normalization (StandardScaler)
 ↓
CNN (Conv1D)
 ↓
Emotion Classification
```

## Model Architecture
The architecture implemented is a 1D Convolutional Neural Network suitable for sequence data. It includes:
- Multiple `Conv1D` layers with ReLU activation.
- `BatchNormalization` for faster and stable convergence.
- `MaxPooling1D` for downsampling.
- `GlobalAveragePooling1D` before the dense layers.
- `Dropout` layers to prevent overfitting.
- A final `Dense` layer with `softmax` activation for multi-class classification.

## Evaluation Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision**: How many of the predicted positives are actual positives.
- **Recall**: How many of the actual positives are predicted as positives.
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visualizes correct predictions and common misclassifications.

## Results
**Final Selected Model (Baseline Architecture): Conv1D**
- **Test Accuracy**: 0.6493
- **Test Precision**: 0.6660
- **Test Recall**: 0.6493
- **Test F1-Score**: 0.6425

*(Note: Validation metrics tracked during training correspond to this exact test set.)*

## Limitations
- Dataset Size: The subset used has 1,440 samples, which is relatively small for deep learning, increasing the risk of overfitting.
- Speaker Variation: Acted speech might differ significantly from natural emotional speech.
- Recording Conditions: Clean, noise-free audio does not represent real-world background noise scenarios.
- **Overfitting & Generalization Experiment**: The baseline model exhibits some overfitting on the training data. Regularization (dropout increased to 0.4) and data augmentation (noise and time-shifting) were evaluated in a controlled experiment to improve generalization. The experiment yielded:
  - Improved Test Accuracy: 0.6840
  - Improved Test F1-Score: 0.6818
  
  While the experiment originally seemed to slightly improve upon a specific rerun (due to stochasticity), the baseline architecture was retained as the final selected model because its overall test performance was generally competitive without the added complexity, and the experiment did not provide a definitive, substantial improvement on held-out test data.

## Future Improvements
- Data Augmentation (e.g., adding noise, time-stretching, pitch-shifting) to improve robustness.
- Larger and more diverse speech datasets.
- More complex models like CNN-BiLSTM or self-attention architectures.
- Real-time microphone inference integration.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook Emotion_Recognition_from_Speech.ipynb
```

The notebook will automatically download the dataset to the `data/` directory and execute the full pipeline.

