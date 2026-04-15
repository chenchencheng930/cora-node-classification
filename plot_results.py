import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

best_df = df.groupby("model", as_index=False)["best_test_acc"].max()

plt.figure(figsize=(8, 5))
plt.bar(best_df["model"], best_df["best_test_acc"])
plt.xlabel("Model")
plt.ylabel("Best Test Accuracy")
plt.title("Model Comparison on Cora")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=300)
plt.show()
