import numpy as np
import tensorflow as tf
import time

print("Loading dataset for representative sample...")
X_train_res = np.load('data/processed/X_train_res.npy')

def representative_dataset():
    for i in range(0, min(500, len(X_train_res)), 1):
        sample = X_train_res[i:i+1].astype(np.float32)
        yield [sample]

print("Converting model...")
converter = tf.lite.TFLiteConverter.from_saved_model('model/saved/ids_model.keras')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('model/saved/ids_model_int8.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved INT8 TFLite model to model/saved/ids_model_int8.tflite")
