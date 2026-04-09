import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===== 1. 读取数据 =====
df = pd.read_csv("processed_data.csv")

X = df[["gripper_pos", "direction_num"]].values
y = df["distance"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ===== 2. 模型 =====
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# ===== 3. 训练 =====
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.6f}")

# ===== 4. 保存模型 =====
torch.save(model.state_dict(), "gripper_model.pt")

print("训练完成！")

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    pred = model(X).numpy()

plt.scatter(y.numpy(), pred, s=2)
plt.xlabel("True Distance")
plt.ylabel("Predicted Distance")
plt.title("Prediction vs True")
plt.show()