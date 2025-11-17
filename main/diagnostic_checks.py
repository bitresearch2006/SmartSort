# diagnostic_checks.py
"""
Run 6 diagnostic checks to help debug low accuracy / dataset-model mismatches.

Place this file in the same folder where your project runs (SmartSort/main) and run:
    python diagnostic_checks.py

Checks performed:
1) Print classes, counts, baseline accuracy.
2) Inspect one training batch (shapes, ranges, example label).
3) Show a few predictions on validation set if model exists.
4) Check model build/trainable params (via build_model if available, or loaded model).
5) Compute a confusion matrix on a small subset of val set (200 samples if available).
6) Print sample image file paths for each class (first N per class).
"""

import os
import sys
import traceback
from pathlib import Path
import numpy as np

# Try imports and guard failures
try:
    from data_prep import (
        CLASS_NAMES,
        IMG_SIZE,
        load_and_prepare_data,
    )
except Exception as e:
    print("ERROR: Could not import data_prep symbols. Make sure data_prep.py is available and in PYTHONPATH.")
    print("Import error:", e)
    traceback.print_exc()
    sys.exit(1)

# Helper: count images per class by walking dataset folder if possible
def count_images_by_class(dataset_dir):
    dataset_dir = Path(dataset_dir)
    counts = {}
    if not dataset_dir.exists():
        return counts
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir():
            cnt = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"])
            counts[d.name] = cnt
    return counts

# 1) Classes, counts, baseline
def check_classes_and_counts():
    print("\n=== CHECK 1: Classes, counts, baseline ===")
    try:
        classes = list(CLASS_NAMES)
    except Exception:
        classes = []
    print("NUM_CLASSES:", len(classes))
    print("CLASS_NAMES:", classes)
    # Attempt to fetch dataset dir for counts - common defaults
    guessed_paths = [
        os.environ.get("BIOWASTE_DATASET"),
        "../data/biomedical_dataset",
        "data/biomedical_dataset",
        "../data",
        "data"
    ]
    found = None
    for p in guessed_paths:
        if p:
            pth = Path(p)
            if pth.exists() and any(pth.iterdir()):
                found = pth
                break
    if found:
        counts = count_images_by_class(found)
        if counts:
            print("Image counts by folder (sample):")
            for k in sorted(counts.keys()):
                print(f"  {k}: {counts[k]}")
        else:
            print(f"No per-class image folders found under guessed path: {found}")
    else:
        print("Dataset folder not found among guessed paths; set BIOWASTE_DATASET env var or check location.")
    if len(classes) > 0:
        baseline = 1.0 / len(classes)
        print(f"Random baseline accuracy ≈ {baseline:.4f} ({baseline*100:.2f}%)")

# 2) Inspect a batch
def check_batch_sample():
    print("\n=== CHECK 2: Inspect one training batch ===")
    try:
        train_ds, val_ds, test_ds, classes = load_and_prepare_data()
    except TypeError:
        # older loaders might return 3 items
        try:
            train_ds, val_ds, test_ds = load_and_prepare_data()
            classes = list(CLASS_NAMES)
        except Exception as e:
            print("Failed to call load_and_prepare_data():", e)
            traceback.print_exc()
            return

    if train_ds is None:
        print("train_ds is None (no training data).")
        return
    try:
        for x, y in train_ds.take(1):
            x_np = x.numpy()
            print("X shape:", x_np.shape)
            print("X dtype:", x_np.dtype)
            print("X min/max:", float(x_np.min()), float(x_np.max()))
            # Labels: could be one-hot or scalar ints
            y_np = y.numpy()
            print("Y shape:", y_np.shape)
            # show first label sample
            first = y_np[0]
            print("First label sample (raw):", first)
            # If one-hot, show argmax
            if first.ndim > 0:
                print("First label argmax:", int(np.argmax(first)))
            break
    except Exception as e:
        print("Error while sampling a batch from train_ds:", e)
        traceback.print_exc()

# 3) Quick predict sanity on a few val images
def check_val_predictions(sample_n=5, model_path="trashnet_classifier.keras"):
    print("\n=== CHECK 3: Run a few predictions on validation set ===")
    import tensorflow as tf
    if not Path(model_path).exists():
        print(f"Model file not found at '{model_path}'. Skipping predictions check.")
        return
    try:
        model = tf.keras.models.load_model(model_path)
        print("Loaded model:", model_path)
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
        return

    # load validation dataset
    try:
        train_ds, val_ds, test_ds, classes = load_and_prepare_data()
    except TypeError:
        train_ds, val_ds, test_ds = load_and_prepare_data()
        classes = list(CLASS_NAMES)

    if val_ds is None:
        print("val_ds is None (no validation data).")
        return

    try:
        i = 0
        for batch in val_ds.unbatch().batch(1).take(sample_n):
            x, y = batch
            p = model.predict(x, verbose=0)[0]
            p = np.asarray(p, dtype=float)
            print(f"\nSample {i+1}: pred_sum={p.sum():.4f} max={p.max():.4f} argmax={int(p.argmax())}")
            # show top 5 probabilities
            top_idx = np.argsort(p)[-5:][::-1]
            print(" Top probs (idx:prob):", [(int(idx), float(np.round(p[idx],4))) for idx in top_idx])
            # show true label info
            y_np = y.numpy()
            if y_np.ndim == 1:
                # one-hot or logits
                true_idx = int(np.argmax(y_np))
                print(" True label (argmax):", true_idx)
            else:
                try:
                    true_idx = int(y_np[0])
                    print(" True label (int):", true_idx)
                except Exception:
                    print(" True label raw:", y_np)
            i += 1
    except Exception as e:
        print("Error during val prediction loop:", e)
        traceback.print_exc()

# 4) Check model build / trainable params
def check_model_build_and_trainable():
    print("\n=== CHECK 4: Model build / trainable params ===")
    try:
        # Try to import build_model from train_model.py if present
        from train_model import build_model
        build_available = True
    except Exception:
        build_available = False

    import tensorflow as tf
    if build_available:
        try:
            _, _, _, classes = load_and_prepare_data()
        except Exception:
            _, _, _ = load_and_prepare_data()
            classes = list(CLASS_NAMES)
        try:
            m = build_model(len(classes))
            # force build
            m.build((None, IMG_SIZE[0], IMG_SIZE[1], 3))
            m.summary()
            trainable = np.sum([tf.keras.backend.count_params(w) for w in m.trainable_weights])
            total = np.sum([tf.keras.backend.count_params(w) for w in m.weights])
            print("Trainable params:", trainable)
            print("Total params:", total)
        except Exception as e:
            print("Error building model via build_model():", e)
            traceback.print_exc()
    else:
        print("build_model() not available in train_model.py. Trying to load saved model if present.")
        mpath = Path("trashnet_classifier.keras")
        if not mpath.exists():
            print("No saved model found at 'trashnet_classifier.keras'. Skipping.")
            return
        try:
            m = tf.keras.models.load_model(str(mpath))
            m.summary()
            trainable = np.sum([tf.keras.backend.count_params(w) for w in m.trainable_weights])
            total = np.sum([tf.keras.backend.count_params(w) for w in m.weights])
            print("Trainable params:", trainable)
            print("Total params:", total)
        except Exception as e:
            print("Error loading model:", e)
            traceback.print_exc()

# 5) Compute confusion matrix (small subset)
def compute_confusion_matrix(max_samples=200, model_path="trashnet_classifier.keras"):
    print("\n=== CHECK 5: Confusion matrix on small validation subset ===")
    import tensorflow as tf
    if not Path(model_path).exists():
        print(f"No model found at {model_path}. Skipping confusion matrix.")
        return
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
        return

    try:
        train_ds, val_ds, test_ds, classes = load_and_prepare_data()
    except TypeError:
        train_ds, val_ds, test_ds = load_and_prepare_data()
        classes = list(CLASS_NAMES)

    if val_ds is None:
        print("No val dataset available.")
        return

    y_true = []
    y_pred = []
    taken = 0
    for batch in val_ds.unbatch().batch(1).take(max_samples):
        x, y = batch
        p = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(p))
        # determine true index for one-hot or int labels
        y_np = y.numpy()
        if y_np.ndim > 1:
            true_idx = int(np.argmax(y_np[0]))
        else:
            try:
                true_idx = int(y_np[0])
            except Exception:
                true_idx = int(np.argmax(y_np[0]))
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        taken += 1

    if taken == 0:
        print("No samples taken from validation set (empty?).")
        return

    # compute confusion matrix without sklearn
    K = max(max(y_true), max(y_pred)) + 1
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print(f"Computed confusion matrix on {taken} samples. Shape: {cm.shape}")
    print(cm)
    # save for later inspection
    np.save("confusion_matrix.npy", cm)
    print("Saved confusion_matrix.npy in current folder.")

# 6) Print sample image paths per class
def show_sample_paths_per_class(dataset_dir_candidate_list=None, max_per_class=5):
    print("\n=== CHECK 6: Sample image paths per class ===")
    if dataset_dir_candidate_list is None:
        dataset_dir_candidate_list = [
            os.environ.get("BIOWASTE_DATASET"),
            "../data/biomedical_dataset",
            "data/biomedical_dataset",
            "../data",
            "data"
        ]
    chosen = None
    for p in dataset_dir_candidate_list:
        if not p:
            continue
        pth = Path(p)
        if pth.exists():
            chosen = pth
            break
    if chosen is None:
        print("Could not guess dataset directory. Set BIOWASTE_DATASET env var or run script from a different working dir.")
        return

    print("Using dataset root:", chosen)
    for cname in CLASS_NAMES:
        cdir = chosen / cname
        if not cdir.exists():
            print(f"  Class folder missing: {cdir} (skipping)")
            continue
        imgs = sorted([str(x) for x in cdir.rglob("*") if x.is_file() and x.suffix.lower() in [".jpg", ".jpeg", ".png"]])[:max_per_class]
        print(f"  {cname}: {len(imgs)} samples (showing up to {max_per_class}):")
        for im in imgs:
            print("    ", im)

# Run all checks
if __name__ == "__main__":
    try:
        check_classes_and_counts()
    except Exception:
        print("CHECK 1 failed unexpectedly.")
        traceback.print_exc()

    try:
        check_batch_sample()
    except Exception:
        print("CHECK 2 failed unexpectedly.")
        traceback.print_exc()

    try:
        check_val_predictions()
    except Exception:
        print("CHECK 3 failed unexpectedly.")
        traceback.print_exc()

    try:
        check_model_build_and_trainable()
    except Exception:
        print("CHECK 4 failed unexpectedly.")
        traceback.print_exc()

    try:
        compute_confusion_matrix()
    except Exception:
        print("CHECK 5 failed unexpectedly.")
        traceback.print_exc()

    try:
        show_sample_paths_per_class()
    except Exception:
        print("CHECK 6 failed unexpectedly.")
        traceback.print_exc()

    print("\nDiagnostics complete. Inspect printed output and 'confusion_matrix.npy' if created.")
