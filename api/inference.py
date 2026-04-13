import numpy as np
import tensorflow as tf
import time

class IDSInference:
    def __init__(self, model_path: str):
        # We use tensorflow's tflite directly to avoid specific dependency issues with tflite-runtime in typical venvs
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, sample: np.ndarray) -> tuple[int, float]:
        """Returns (class_index, latency_ms)."""
        # Convert FP32 scaled array to INT8 format since model expects INT8
        # The input scale and zero_point from input details must be used
        input_details = self.input_details[0]
        scale, zero_point = input_details['quantization']
        
        # Original features scaled standard float32
        inp = sample.reshape(1, 1, -1)
        
        if scale > 0.0:
            inp = inp / scale + zero_point
            inp = np.round(inp).astype(input_details['dtype'])
        else:
            inp = inp.astype(np.float32) # fallback if model not actually int8 input

        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]['index'])
        latency_ms = (time.perf_counter() - start) * 1000

        return int(raw.argmax()), latency_ms
