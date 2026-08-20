# ==============================================================================
# DISEASE PREDICTION MODEL -> TFLITE EXPORT PIPELINE
# (Replaces the sklearn ensemble with a Keras model so it can be converted
#  to .tflite for the Flutter app. sklearn/XGBoost models CANNOT be
#  converted to TFLite directly — this is why the app was failing to load.)
# ==============================================================================

import pandas as pd
import numpy as np
import json
import os
import glob

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------------------------

def find_file(filename):
    matches = glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
    if not matches:
        all_files = glob.glob('/kaggle/input/**/*', recursive=True)
        for f in all_files:
            if filename.lower() in os.path.basename(f).lower():
                return f
        raise FileNotFoundError(f"Could not find {filename} in /kaggle/input/")
    return matches[0]

dataset_path = find_file('DiseaseAndSymptoms.csv') if glob.glob('/kaggle/input/**/DiseaseAndSymptoms.csv', recursive=True) else find_file('dataset.csv')
precaution_path = find_file('Disease precaution.csv') if glob.glob('/kaggle/input/**/Disease precaution.csv', recursive=True) else find_file('symptom_precaution.csv')

print(f"Dataset path found: {dataset_path}")
print(f"Precaution path found: {precaution_path}")

symptoms_df = pd.read_csv(dataset_path)
precautions_df = pd.read_csv(precaution_path)

# ------------------------------------------------------------------------------
# 2. FEATURE EXTRACTION & ONE-HOT ENCODING
# ------------------------------------------------------------------------------
print("Preprocessing symptoms into binary feature vectors...")

symptom_cols = [f'Symptom_{i}' for i in range(1, 18) if f'Symptom_{i}' in symptoms_df.columns]
all_symptoms = set()

for col in symptom_cols:
    unique_in_col = [str(s).strip() for s in symptoms_df[col].dropna().unique()]
    all_symptoms.update(unique_in_col)

symptom_list = sorted(list(all_symptoms))
print(f"Total Unique Diseases: {symptoms_df['Disease'].nunique()}")
print(f"Total Unique Symptoms Extracted: {len(symptom_list)}")

encoded_rows = []
for idx, row in symptoms_df.iterrows():
    present_symptoms = set(str(s).strip() for s in row[symptom_cols].dropna().values)
    row_dict = {symptom: (1 if symptom in present_symptoms else 0) for symptom in symptom_list}
    row_dict['Disease'] = row['Disease']
    encoded_rows.append(row_dict)

dataset_processed = pd.DataFrame(encoded_rows).drop_duplicates().reset_index(drop=True)
print(f"Rows after dedup: {len(dataset_processed)}")

X = dataset_processed.drop(columns=['Disease']).values.astype('float32')
y_raw = dataset_processed['Disease']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
num_features = X.shape[1]

class_counts = pd.Series(y).value_counts()
use_stratify = class_counts.min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if use_stratify else None
)

# ------------------------------------------------------------------------------
# 3. BUILD & TRAIN A KERAS MODEL (this is what actually converts to TFLite)
# ------------------------------------------------------------------------------
print(f"\nBuilding Keras model: {num_features} inputs -> {num_classes} disease classes")

model = keras.Sequential([
    keras.layers.Input(shape=(num_features,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=16,
    verbose=2
)

y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n==========================================")
print(f" KERAS MODEL TEST ACCURACY: {acc * 100:.2f}%")
print(f"==========================================\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# ------------------------------------------------------------------------------
# 4. CONVERT TO TFLITE (this is the file the Flutter app actually loads)
# ------------------------------------------------------------------------------
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # shrinks file size for mobile
tflite_model = converter.convert()

with open('disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Saved disease_model.tflite ({len(tflite_model) / 1024:.1f} KB)")
print(f"Input shape the app must send: [1, {num_features}] float32")
print(f"Output shape the app will receive: [1, {num_classes}] float32 (softmax probabilities)")

# ------------------------------------------------------------------------------
# 5. EXPORT SCHEMA & PRECAUTIONS (unchanged format, still needed by the app)
# ------------------------------------------------------------------------------
symptom_mapping = {
    "symptoms": symptom_list,
    "diseases": list(label_encoder.classes_)
}
with open('symptom_schema.json', 'w') as f:
    json.dump(symptom_mapping, f, indent=4)

precaution_cols = [c for c in precautions_df.columns if 'precaution' in c.lower()]
precaution_map = {}
for _, row in precautions_df.iterrows():
    d_name = row['Disease']
    p_list = [str(row[c]).title() for c in precaution_cols if pd.notna(row[c]) and str(row[c]) != 'nan']
    precaution_map[d_name] = p_list

with open('disease_precautions.json', 'w') as f:
    json.dump(precaution_map, f, indent=4)

print("\nAll assets saved:")
print(" - disease_model.tflite      <- copy to flutter_app/assets/")
print(" - symptom_schema.json       <- copy to flutter_app/assets/")
print(" - disease_precautions.json  <- copy to flutter_app/assets/")
print("\nNOTE: Every disease name in symptom_schema.json['diseases'] should match")
print("a key in disease_precautions.json. Any mismatch just means that disease")
print("won't have precautions shown (app already handles this gracefully).")
