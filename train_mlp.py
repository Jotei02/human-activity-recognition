from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import os
import numpy as np
from model import MLP

os.makedirs("model", exist_ok=True)   # fresh clone has no model/ yet

X_train = torch.from_numpy(np.load("data/processed/X_train.npy")).float()
y_train = torch.from_numpy(np.load("data/processed/y_train.npy")).long()

scaler = StandardScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train.numpy())).float()
joblib.dump(scaler, "model/scaler.pkl")

y_train = y_train.squeeze() - 1  # 1-indexed → 0-indexed

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MLP(input_dim=X_train.shape[1], hidden_dim=128, num_classes=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

torch.save(model.state_dict(), "model/mlp.pth")

# ------------------------------------------------------------------
# Export to ONNX
# ------------------------------------------------------------------

model.eval()  # must be in eval mode before export — disables dropout/batchnorm if any

# FIX 1: batch size 2, not 1.
# Batch=1 lets constant folding bake the shape into the graph,
# which breaks the quantizer's shape inference later.
dummy_input = torch.randn(2, X_train.shape[1])

torch.onnx.export(
    model,
    dummy_input,
    "model/mlp_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    do_constant_folding=False,  # FIX 2: prevents freezing intermediate shapes
    opset_version=17,           # FIX 3: better ORT quantizer coverage than 18
    export_params=True,
)

import onnx
_m = onnx.load("model/mlp_model.onnx")          # also reads the .data file if one was written
onnx.save_model(_m, "model/mlp_model.onnx", save_as_external_data=False)
if os.path.exists("model/mlp_model.onnx.data"):
    os.remove("model/mlp_model.onnx.data")

print("Model training and export completed.")

# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

import onnxruntime as ort

sess = ort.InferenceSession("model/mlp_model.onnx")
test_input = np.random.randn(4, X_train.shape[1]).astype(np.float32)
out = sess.run(None, {"input": test_input})
print(f"ONNX output shape: {out[0].shape}")  # should be (4, 6)
