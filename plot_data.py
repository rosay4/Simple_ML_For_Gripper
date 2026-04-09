import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("processed_data.csv")

# 按 direction 分开
df_open = df[df["direction_num"] == 1]
df_close = df[df["direction_num"] == 0]

# 画图
plt.figure(figsize=(8,6))

plt.scatter(df_open["gripper_pos"], df_open["distance"], s=2, label="open")
plt.scatter(df_close["gripper_pos"], df_close["distance"], s=2, label="close")

plt.xlabel("gripper_pos")
plt.ylabel("distance")
plt.legend()
plt.title("Gripper Mapping")

plt.show()