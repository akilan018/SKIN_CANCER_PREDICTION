import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Load model
model = tf.keras.models.load_model("best_model.h5", compile=False)

print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(f"\nModel Type: {type(model).__name__}")
print(f"Total Layers: {len(model.layers)}")
print(f"Input Shape: {model.input_shape}")
print(f"Output Shape: {model.output_shape}")

# Count parameters
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print(f"\nTotal params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Non-trainable params: {non_trainable_params:,}")

print("\n" + "=" * 60)
print("FIRST FEW LAYERS")
print("=" * 60)
for i, layer in enumerate(model.layers[:5]):
    print(f"{i+1}. {layer.__class__.__name__}: {layer.name}")

print("\n" + "=" * 60)
print("LAST FEW LAYERS")
print("=" * 60)
for i, layer in enumerate(model.layers[-5:]):
    print(f"{len(model.layers)-4+i}. {layer.__class__.__name__}: {layer.name}")
