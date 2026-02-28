import json
import pandas as pd
import matplotlib.pyplot as plt


with open("outputs/final_model_cpu/training_stats.json", "r") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# training updates
train_rows = df[df["loss"].notnull()] if "loss" in df.columns else pd.DataFrame()

# final summary rows
summary_rows = df[df["train_loss"].notnull()] if "train_loss" in df.columns else pd.DataFrame()

plt.figure(figsize=(8,5))

if len(train_rows) > 0:
    plt.plot(train_rows["step"], train_rows["loss"], label="train loss")

if len(summary_rows) > 0:
    plt.scatter(summary_rows["step"], summary_rows["train_loss"], label="train loss summary")

plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.title("learning curve")
plt.show()
