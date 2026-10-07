import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# 1. Project Title
cells.append(nbf.v4.new_markdown_cell("# Emotion Recognition from Speech\n\n**CodeAlpha Internship Task 2**"))

# 2. Objective
cells.append(nbf.v4.new_markdown_cell("## Objective\nThe objective of this project is to build a machine-learning/deep-learning system that recognizes human emotions from speech audio using MFCC features and a neural network classifier. We will use the RAVDESS dataset, extract features using `librosa`, and train a Convolutional Neural Network (CNN) to classify the emotions."))

# 3. Import Libraries
cells.append(nbf.v4.new_markdown_cell("## Import Libraries"))
cells.append(nbf.v4.new_code_cell("""import os
import zipfile
import urllib.request
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from IPython.display import Audio
"""))

# 4. Dataset
cells.append(nbf.v4.new_markdown_cell("## Dataset\nWe are using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.\n\n"
"Emotions encoded as:\n"
"01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.\n"
"The script below will download the dataset automatically using `kagglehub`."))
cells.append(nbf.v4.new_code_cell("""# Download RAVDESS Audio Speech Dataset using kagglehub
import kagglehub
import shutil

EXTRACT_DIR = "data/raw"
os.makedirs("data", exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

if not os.listdir(EXTRACT_DIR):
    print("Downloading RAVDESS dataset via kagglehub...")
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    print("Download complete. Copying to data/raw...")
    # Copy files to our project directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(EXTRACT_DIR, file)
                shutil.copy2(src_path, dest_path)
    print("Dataset ready in data/raw/")
else:
    print("Dataset already extracted in data/raw/")
"""))

# 5. Audio Exploration
cells.append(nbf.v4.new_markdown_cell("## Audio Exploration\nLet's load a sample audio file and visualize its waveform and spectrogram."))
cells.append(nbf.v4.new_code_cell("""sample_files = glob.glob(f"{EXTRACT_DIR}/**/*.wav", recursive=True)
if len(sample_files) > 0:
    sample_file = sample_files[0]
    print(f"Sample file: {sample_file}")
    
    # Load audio
    data, sample_rate = librosa.load(sample_file, sr=None)
    
    # Display waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(data, sr=sample_rate)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    
    # Display spectrogram
    plt.figure(figsize=(12, 4))
    D = librosa.stft(data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.show()
    
else:
    print("No audio files found. Please check dataset extraction.")
"""))

# 6. Emotion Labels
cells.append(nbf.v4.new_markdown_cell("## Emotion Labels\nThe RAVDESS filename has a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav).\nThe 3rd part is the emotion label.\n\n"
"Emotions:\n"
"01: Neutral\n02: Calm\n03: Happy\n04: Sad\n05: Angry\n06: Fearful\n07: Disgust\n08: Surprised\n"))
cells.append(nbf.v4.new_code_cell("""emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
"""))

# 7. MFCC Feature Extraction
cells.append(nbf.v4.new_markdown_cell("## MFCC Feature Extraction\nWe will extract 40 MFCC coefficients from each audio file. MFCCs capture the power spectrum of the audio."))
cells.append(nbf.v4.new_code_cell("""def extract_mfcc(file_path, n_mfcc=40):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=22050)
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # We take the mean across the time axis so we have a fixed length feature vector per audio
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None
"""))

# 8. Dataset Preparation
cells.append(nbf.v4.new_markdown_cell("## Dataset Preparation\nWe will iterate through all audio files, extract features and labels, and prepare our X (features) and y (labels). We then split them into training and testing sets."))
cells.append(nbf.v4.new_code_cell("""X, y = [], []

all_files = glob.glob(f"{EXTRACT_DIR}/**/*.wav", recursive=True)
print(f"Total files found: {len(all_files)}")

for file in all_files:
    filename = os.path.basename(file)
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        if emotion_code in emotion_dict:
            features = extract_mfcc(file, n_mfcc=40)
            if features is not None:
                X.append(features)
                y.append(emotion_code)

X = np.array(X)
y = np.array(y)
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# Encode labels to integers
unique_labels = sorted(list(set(y)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

y_encoded = np.array([label_to_int[label] for label in y])

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
"""))

# 9. Feature Normalization
cells.append(nbf.v4.new_markdown_cell("## Feature Normalization\nStandardize features. It is important to fit the scaler ONLY on the training data to avoid data leakage."))
cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for Conv1D input
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
print(f"Reshaped X_train: {X_train.shape}")
print(f"Reshaped X_test: {X_test.shape}")
"""))

# 10. Neural Network Model
cells.append(nbf.v4.new_markdown_cell("## Neural Network Model\nWe build a Deep Learning classifier using Conv1D layers which are suitable for sequence data like MFCCs."))
cells.append(nbf.v4.new_code_cell("""def build_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv1D(64, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, kernel_size=5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Dropout(0.3))
    
    model.add(Conv1D(256, kernel_size=5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

num_classes = len(unique_labels)
model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)
model.summary()
"""))

# 11. Model Compilation
cells.append(nbf.v4.new_markdown_cell("## Model Compilation"))
cells.append(nbf.v4.new_code_cell("""model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
"""))

# 12. Training
cells.append(nbf.v4.new_markdown_cell("## Training"))
cells.append(nbf.v4.new_code_cell("""early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

os.makedirs("results", exist_ok=True)

history = model.fit(X_train, y_train, 
                    epochs=70, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, lr_reducer],
                    verbose=1)
"""))

# 13. Training History
cells.append(nbf.v4.new_markdown_cell("## Training History"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/training_history.png')
plt.show()
"""))

# 14. Evaluation
cells.append(nbf.v4.new_markdown_cell("## Evaluation"))
cells.append(nbf.v4.new_code_cell("""test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

target_names = [emotion_dict[int_to_label[i]] for i in range(num_classes)]
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
"""))

# 15. Confusion Matrix
cells.append(nbf.v4.new_markdown_cell("## Confusion Matrix"))
cells.append(nbf.v4.new_code_cell("""cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('results/confusion_matrix.png')
plt.show()
"""))

# 16. Sample Predictions
cells.append(nbf.v4.new_markdown_cell("## Sample Predictions"))
cells.append(nbf.v4.new_code_cell("""num_samples = 5
indices = np.random.choice(len(y_test), num_samples, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    actual_emotion = emotion_dict[int_to_label[y_test[idx]]]
    predicted_emotion = emotion_dict[int_to_label[y_pred[idx]]]
    confidence = np.max(y_pred_probs[idx]) * 100
    
    text = f"Actual: {actual_emotion}\\nPred: {predicted_emotion}\\nConf: {confidence:.1f}%"
    
    plt.subplot(1, num_samples, i+1)
    plt.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.axis('off')
    
plt.suptitle('Sample Predictions', fontsize=16)
plt.tight_layout()
plt.savefig('results/sample_predictions.png')
plt.show()
"""))

# 17. Model Saving
cells.append(nbf.v4.new_markdown_cell("## Model Saving"))
cells.append(nbf.v4.new_code_cell("""os.makedirs("models", exist_ok=True)
model_path = 'models/best_emotion_model.keras'
model.save(model_path)
print(f"Model saved to {model_path}")

loaded_model = tf.keras.models.load_model(model_path)
verify_pred_probs = loaded_model.predict(X_test[:1])
verify_pred = np.argmax(verify_pred_probs, axis=1)
print(f"Verification prediction: {emotion_dict[int_to_label[verify_pred[0]]]} (Actual: {emotion_dict[int_to_label[y_test[0]]]})")
"""))

# 18. Results
cells.append(nbf.v4.new_markdown_cell("## Results"))
cells.append(nbf.v4.new_code_cell("""with open('results/metrics.txt', 'w') as f:
    f.write("Emotion Recognition from Speech\\n")
    f.write("Model: Conv1D\\n\\n")
    f.write(f"Accuracy: {acc:.4f}\\n")
    f.write(f"Precision: {prec:.4f}\\n")
    f.write(f"Recall: {rec:.4f}\\n")
    f.write(f"F1-Score: {f1:.4f}\\n")
print("Metrics saved to results/metrics.txt")
"""))

# 19. Conclusion
cells.append(nbf.v4.new_markdown_cell("## Conclusion\nWe successfully built an Emotion Recognition model from speech audio using the RAVDESS dataset. MFCC features were extracted and a Conv1D neural network was trained. The model achieves reasonable accuracy, and further improvements could involve data augmentation, more complex architectures (like CNN-BiLSTM), or using a larger dataset."))

nb.cells = cells
with open("Emotion_Recognition_from_Speech.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook Emotion_Recognition_from_Speech.ipynb created successfully.")

