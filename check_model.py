import h5py

with h5py.File('best_model.h5', 'r') as f:
    keras_version = f.attrs.get('keras_version', 'unknown')
    backend = f.attrs.get('backend', 'unknown')
    print(f"Keras version: {keras_version}")
    print(f"Backend: {backend}")
