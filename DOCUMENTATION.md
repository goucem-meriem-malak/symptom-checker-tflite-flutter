# Documentation — AI Symptom Checker (TFLite + Flutter)

Full technical walkthrough of the methodology, decisions, and architecture. See `README.md` for the quick overview.

---

## 1. Problem framing

Build a mobile app where a user selects symptoms and receives a predicted condition, running entirely on-device (no server round-trip). This means the trained model has to end up in a mobile-compatible format (TFLite), not just a Python `.pkl`/`.h5` file.

---

## 2. Data pipeline (`model_training/train_and_export_tflite.py`)

### 2.1 Source data
Two CSVs: a disease/symptom mapping table (up to 17 symptom columns per disease row) and a disease/precaution mapping table (up to 4 precautions per disease).

### 2.2 Feature engineering
Symptoms are converted from a wide, sparse, positional format (`Symptom_1`, `Symptom_2`, ... `Symptom_17`, with nulls where a disease has fewer symptoms) into a **binary one-hot matrix**: one column per unique symptom across the whole dataset, 1 if present for that disease, 0 otherwise.

```python
present_symptoms = set(row[symptom_cols].dropna().values)
row_dict = {symptom: (1 if symptom in present_symptoms else 0) for symptom in symptom_list}
```

This produces a fixed-width feature vector (131 symptoms in this run) regardless of how many symptoms any individual disease row originally listed — necessary because the mobile app needs a consistent input shape for the model.

### 2.3 Deduplication
Exact duplicate rows are dropped before splitting. Without this, identical rows could land in both train and test, inflating the reported accuracy without the model actually generalizing better.

### 2.4 Label conflict check
Before training, the pipeline checks whether any symptom combination maps to more than one disease:
```python
conflict_groups = dataset_processed.groupby(list(X_check.columns))['Disease'].nunique()
```
If this is non-zero, it represents a hard accuracy ceiling — no model can distinguish two diseases with identical symptom vectors. This dataset has none, meaning 100% test accuracy is mathematically achievable, though not necessarily meaningful outside this synthetic-ish, deterministic dataset (see Limitations).

---

## 3. Model architecture

### 3.1 Why Keras, not scikit-learn/XGBoost

The first version of this project used an sklearn `VotingClassifier` (RandomForest + ExtraTrees + XGBoost), chosen for strong tabular accuracy. **This was a dead end for mobile deployment**: sklearn and XGBoost models have no supported conversion path to TFLite. `joblib.dump()` produces a `.pkl` file that a Flutter/TFLite runtime cannot load under any circumstances.

Switching to a Keras `Sequential` model was necessary specifically because `tf.lite.TFLiteConverter.from_keras_model()` is a first-class, supported conversion path. This is a hard constraint to know before picking a training library on any project with mobile deployment as a goal — not a stylistic preference.

### 3.2 Network
```
Input(131 features)
  → Dense(256, relu) → Dropout(0.3)
  → Dense(128, relu) → Dropout(0.2)
  → Dense(41, softmax)
```
Simple feedforward network — appropriate given the input is already a clean binary feature vector (no spatial or sequential structure to exploit), and the dataset is small enough that a deeper/larger network would risk overfitting without benefit.

### 3.3 TFLite conversion — a real deployment bug and its fix

**First export attempt** used:
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # post-training quantization
```

This produced a smaller file but caused a runtime crash on-device:
```
E/tflite: Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'.
E/tflite: Registration failed.
I/flutter: Initialization Error: Invalid argument(s): Unable to create interpreter.
```

**Root cause**: quantization changes the op version the converter emits for `FULLY_CONNECTED`. The version emitted (12) was newer than what the `tflite_flutter` plugin's bundled TFLite runtime on the test device supported.

**Fix**: remove the `optimizations` line, exporting plain float32:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()  # no quantization
```

**Trade-off accepted**: larger file size (~283 KB vs. a smaller quantized version) in exchange for broad runtime compatibility. At this model's small size, the trade-off is a non-issue for app size budgets.

**General lesson**: quantization is not free — it can silently require a newer runtime than your target devices ship with. Test on-device after any conversion setting change, not just in the Python conversion step.

---

## 4. Flutter app architecture (`flutter_app/lib/main.dart`)

### 4.1 Startup sequence (`_initializeModelAndSchema`)
1. Load `symptom_schema.json` from assets → gives the exact ordered list of symptom names (131) and disease labels (41) the model expects.
2. Load `disease_precautions.json` from assets → disease name → list of precaution strings.
3. Load `disease_model.tflite` via `Interpreter.fromAsset`.
4. Validate that the model's actual input tensor shape matches the schema's symptom count — logs a warning if they diverge (e.g. after retraining with a different feature set without re-copying the schema).

If any of these three assets fail to load, the app shows an explicit error screen with the underlying exception message, rather than hanging on the loading spinner indefinitely — asset/model loading failures are the most common first-run issue with TFLite mobile apps, so surfacing the error is deliberate.

### 4.2 Inference (`_predictDisease`)
1. Build a 131-length float vector: 1.0 for each symptom the user selected (in schema order), 0.0 otherwise.
2. Reshape to `[1, 131]` (batch dimension of 1) and run through the interpreter.
3. Output is `[1, 41]` softmax probabilities — argmax gives the predicted disease index, confidence is the max probability.
4. Look up precautions for the predicted disease name directly from the loaded JSON map (falls back to generic guidance text if a disease name isn't found in the precautions map — handles any mismatch between the two source files gracefully rather than crashing).
5. Navigates to `DiagnosisDetailScreen` with the prediction, confidence, and precautions.

### 4.3 Why precautions moved from a hardcoded map to the JSON file
An earlier version of the app had a hardcoded Dart `Map<String, DiseaseInfo>` covering only 5 of the ~40 diseases the model could predict — meaning most predictions showed no precaution info at all, and the JSON file generated by the training pipeline was never actually used. Loading `disease_precautions.json` at runtime instead means every disease the model can predict has precaution text, sourced from the same pipeline that trained the model, with no manual per-disease maintenance required in the app code.

---

## 5. Known limitations (stated deliberately, not hidden)

- The training dataset maps symptoms to diseases close to deterministically, with very little of the ambiguity real-world symptom reporting has. High test accuracy here reflects the dataset's clean structure more than genuine diagnostic difficulty — this should not be read as "the model is good at diagnosing disease from symptoms" in a real clinical sense.
- Precaution text is dataset-sourced, generic, and not clinically reviewed.
- This is a technical demonstration of on-device ML deployment, not a medical device or diagnostic tool. The app includes an explicit disclaimer for this reason.

---

## 6. File reference

| File | Purpose |
|---|---|
| `model_training/train_and_export_tflite.py` | Full pipeline: load → encode → train Keras model → convert to TFLite → export schema/precautions |
| `model_training/outputs/disease_model.tflite` | Trained model, float32, ~283 KB |
| `model_training/outputs/symptom_schema.json` | Ordered symptom list + disease label list (must match model input/output order) |
| `model_training/outputs/disease_precautions.json` | Disease name → precaution list |
| `flutter_app/lib/main.dart` | App entry point, symptom selector, inference logic, results screen |
| `flutter_app/pubspec.yaml` | Dependencies (`tflite_flutter`) and asset declarations |
| `flutter_app/assets/` | Runtime copy of the three files above — must match `model_training/outputs/` exactly |
