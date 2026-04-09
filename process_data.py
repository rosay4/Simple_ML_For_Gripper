import pandas as pd

# ===== 1. 读取数据 =====
input_file = "gripper_calibration_dataset - Sheet1.csv"   # 你的CSV文件名
output_file = "processed_data.csv"

df = pd.read_csv(input_file)

# ===== 2. 排序（强烈建议）=====
df = df.sort_values(by=["run_id", "step_index"]).reset_index(drop=True)

# ===== 3. 分组平滑（按 run_id）=====
# rolling window = 5（可以调：3 / 5 / 7）
WINDOW_SIZE = 5

df["distance_smooth"] = (
    df.groupby("run_id")["distance"]
    .transform(lambda x: x.rolling(WINDOW_SIZE, center=True).mean())
)

# ===== 4. 处理边界NaN =====
df["distance_smooth"] = df["distance_smooth"].fillna(method="bfill")
df["distance_smooth"] = df["distance_smooth"].fillna(method="ffill")

# ===== 5. （可选）降采样 =====
# 每隔N个点取一个（建议 3~5）
DOWNSAMPLE_STEP = 3

df_downsampled = df.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)

# ===== 6. 选择训练字段 =====
df_final = df_downsampled[
    [
        "gripper_pos",        # 输入
        "direction_num",      # 输入
        "distance_smooth"     # 输出
    ]
]

# 重命名方便训练
df_final = df_final.rename(columns={
    "distance_smooth": "distance"
})

# ===== 7. 保存 =====
df_final.to_csv(output_file, index=False)

print("处理完成！输出文件:", output_file)