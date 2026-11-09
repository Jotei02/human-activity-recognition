import onnxruntime as ort
import numpy as np
import joblib

class ONNXPredictor:
    def __init__(self, model_path, scaler_path):
        self.session = ort.InferenceSession(model_path)
        self.scaler = joblib.load(scaler_path)
        self.input_name = self.session.get_inputs()[0].name
        self.labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

    def predict(self, features):
        scaled = self.scaler.transform(features)
        probs = self.session.run(None, {self.input_name: scaled.astype(np.float32)})[0]
        label = np.argmax(probs, axis=1)[0]
        confidence = probs[0, label]
        return self.labels[label], confidence
