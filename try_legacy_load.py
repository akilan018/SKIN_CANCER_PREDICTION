"""
try_legacy_load.py - Try every possible loading method for best_model.h5
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import numpy as np

# Check the saved keras version
with h5py.File("best_model.h5", "r") as f:
    kv = f.attrs.get('keras_version', 'unknown')
    bk = f.attrs.get('backend', 'unknown')
    print(f"Saved with keras_version={kv}, backend={bk}")

import tensorflow as tf
print(f"Current TF: {tf.__version__}, Keras: {tf.keras.__version__}")

# ─── Try 1: legacy_h5_format directly with rebuild=False trick ───────────────────
print("\n--- Try 1: legacy h5 with sequential workaround ---")
try:
    from keras.src.legacy.saving import legacy_h5_format
    model = legacy_h5_format.load_model_from_hdf5("best_model.h5", compile=False)
    print("SUCCESS via legacy_h5_format!")
except Exception as e:
    print(f"FAIL: {e}")

# ─── Try 2: Use tf.saved_model approach - export first as SavedModel ─────────────
# (Skip - requires loading first)

# ─── Try 3: Use keras 2 compat load ─────────────────────────────────────────────
print("\n--- Try 3: tf.compat.v1 approach ---")
try:
    import tensorflow.compat.v1 as tf1
    with tf1.Session() as sess:
        model = tf.keras.models.load_model("best_model.h5", compile=False)
    print("SUCCESS via tf1 session!")
except Exception as e:
    print(f"FAIL: {e}")

# ─── Try 4: Load just the architecture config and manually fix the SE block ──────
print("\n--- Try 4: Direct config manipulation with SE-block fix ---")
import json

with h5py.File("best_model.h5", "r") as f:
    config = json.loads(f.attrs['model_config'])

# Find all Multiply layers in EfficientNetB3 and check their inbound_nodes
eff_fn_layers = config['config']['layers'][1]['config']['layers']
multiply_layers = [(i, l) for i, l in enumerate(eff_fn_layers) if l['class_name'] == 'Multiply']
print(f"Found {len(multiply_layers)} Multiply layers in EfficientNetB3")
if multiply_layers:
    i, ml = multiply_layers[0]
    print(f"Sample Multiply [{i}] inbound_nodes: {json.dumps(ml.get('inbound_nodes', []))[:400]}")
