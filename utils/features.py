import numpy as np

def extract_features(window):
    # window: list of samples, shape = (window_size, 6)
    arr = np.array(window)
    features = []

    # simple handcrafted features
    features.extend(np.mean(arr, axis=0))
    features.extend(np.std(arr, axis=0))
    features.append(np.sqrt((arr**2).sum(axis=1).mean()))  # RMS magnitude
    return np.array(features).reshape(1, -1)
