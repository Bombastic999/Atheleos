

import os
import pickle
import argparse
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ── Optional: ONNX export (pip install skl2onnx) ────────────────────────────
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  skl2onnx not installed — .onnx export will be skipped.")
    print("   Run: pip install skl2onnx onnxruntime\n")


# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Load & validate CSV ──────────────────────────────────────────────────────
def load_data(exercise: str) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """
    Reads data/<exercise>.csv
    Returns X (features), y (encoded labels), feature_names, label_encoder
    """
    path = os.path.join(DATA_DIR, f"{exercise}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file found at {path}")

    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print(f"  Exercise : {exercise}")
    print(f"  Rows     : {len(df)}")
    print(f"  Labels   : {df['label'].value_counts().to_dict()}")
    print(f"{'='*55}")

    # Validate minimum data
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        if count < 10:
            print(f"  ⚠️  Warning: only {count} samples for '{label}' — model may not generalise well")
            print(f"       Recommended minimum: 30 per class\n")

    if len(label_counts) < 2:
        raise ValueError("Need at least 2 classes (e.g. 'good' and one bad type) to train.")

    # Feature columns = everything except exercise name and label
    feature_cols = [c for c in df.columns if c not in ("exercise", "label", "timestamp")]
    X = df[feature_cols].values.astype(np.float32)

    # Encode string labels → integers
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    print(f"  Features : {feature_cols}")
    print(f"  Classes  : {list(le.classes_)}\n")

    return X, y, feature_cols, le


# ── Train ────────────────────────────────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Builds a Pipeline: StandardScaler → SVM (RBF kernel)
    Scaler is important — angle deltas have different ranges per joint.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True,   # enables predict_proba for confidence scores
                       random_state=42))
    ])

    # Cross-validation gives a more honest accuracy than a single split
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  Cross-val accuracy : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    if cv_scores.mean() < 0.80:
        print("  ⚠️  Accuracy below 80% — consider:")
        print("       • Recording more reps (aim for 50+ per class)")
        print("       • Making bad-form reps more clearly wrong")
        print("       • Checking your phase detector is snapshotting correctly\n")
    else:
        print("  ✅ Good accuracy — model looks healthy\n")

    # Final fit on all data
    pipeline.fit(X, y)
    return pipeline


# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder):
    """Prints confusion matrix and per-class precision/recall."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline_eval = Pipeline(pipeline.steps)  # fresh copy for eval
    pipeline_eval.fit(X_train, y_train)
    y_pred = pipeline_eval.predict(X_test)

    print("  Classification report (held-out 20%):")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    print("  Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_
    header = "          " + "  ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {class_names[i][:8]:>8}  {row_str}")
    print()


# ── Save .pkl ────────────────────────────────────────────────────────────────
def save_pkl(pipeline: Pipeline, label_encoder: LabelEncoder,
             feature_names: list[str], exercise: str):
    """
    Saves the pipeline + label encoder + feature names together.
    Load with: data = pickle.load(open('models/squat.pkl','rb'))
               label = data['label_encoder'].inverse_transform(
                           data['pipeline'].predict([feature_vector])
                       )
    """
    path = os.path.join(MODELS_DIR, f"{exercise}.pkl")
    payload = {
        "pipeline":      pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "exercise":      exercise,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  💾 Saved .pkl  → {path}  ({size_kb:.0f} KB)")


# ── Save .onnx ───────────────────────────────────────────────────────────────
def save_onnx(pipeline: Pipeline, feature_names: list[str], exercise: str):
    """
    Exports the trained pipeline to ONNX format for use in the browser
    with onnxruntime-web. The scaler is included in the export automatically.
    """
    if not ONNX_AVAILABLE:
        print("  ⏭️  Skipping .onnx export (skl2onnx not installed)")
        return

    path = os.path.join(MODELS_DIR, f"{exercise}.onnx")
    n_features = len(feature_names)

    try:
        proto = convert_sklearn(
            pipeline,
            f"{exercise}_svm",
            [("input", FloatTensorType([None, n_features]))]
        )
        with open(path, "wb") as f:
            f.write(proto.SerializeToString())
        size_kb = os.path.getsize(path) / 1024
        print(f"  💾 Saved .onnx → {path}  ({size_kb:.0f} KB)")
        print(f"\n  Copy frontend/models/{exercise}.onnx to your web app.")
    except Exception as e:
        print(f"  ❌ ONNX export failed: {e}")
        print("     The .pkl model still works for Python inference.")


# ── Predict helper (sanity check) ────────────────────────────────────────────
def predict_sample(pipeline: Pipeline, label_encoder: LabelEncoder,
                   feature_vector: list[float]) -> dict:
    """
    Call this to test your model on a single rep's feature vector.

    Example:
        result = predict_sample(pipeline, le, [-84.2, -61.4, -3.1, 2.1, -1.3, 0.8])
        print(result)
        # → {'label': 'good', 'confidence': 0.94, 'all_probs': {...}}
    """
    x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    label_idx  = pipeline.predict(x)[0]
    probs      = pipeline.predict_proba(x)[0]
    label      = label_encoder.inverse_transform([label_idx])[0]
    all_probs  = {cls: float(f"{p:.3f}") for cls, p in
                  zip(label_encoder.classes_, probs)}
    return {
        "label":      label,
        "confidence": float(f"{probs[label_idx]:.3f}"),
        "all_probs":  all_probs
    }


# ── Train one exercise ────────────────────────────────────────────────────────
def train_exercise(exercise: str):
    print(f"\n🏋️  Training: {exercise.upper()}")
    X, y, feature_names, le = load_data(exercise)
    pipeline = train(X, y)
    evaluate(pipeline, X, y, le)
    save_pkl(pipeline, le, feature_names, exercise)
    save_onnx(pipeline, feature_names, exercise)
    print(f"\n✅ Done: {exercise}\n")
    return pipeline, le


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train posture SVM classifiers")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exercise", type=str,
                       help="Train one exercise, e.g. --exercise squat")
    group.add_argument("--all", action="store_true",
                       help="Train all exercises found in data/ folder")
    args = parser.parse_args()

    if args.all:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV files found in {DATA_DIR}/")
        else:
            for csv_file in csv_files:
                exercise_name = csv_file.replace(".csv", "")
                try:
                    train_exercise(exercise_name)
                except Exception as e:
                    print(f"❌ Failed to train {exercise_name}: {e}\n")
    else:
        train_exercise(args.exercise)
