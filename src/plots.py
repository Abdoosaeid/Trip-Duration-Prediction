import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot():
    data_path = r"D:\Trip-Duration-Prediction\reports\r2_results.csv"
    save_path = r"D:\Trip-Duration-Prediction\reports\figures\r2_comparison.png"

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Melt the data
    df_melted = df.melt(
        id_vars="Model",
        value_vars=["train", "val"],
        var_name="Dataset",
        value_name="R2"
    )

    # Plot setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    # Barplot
    sns.barplot(
        data=df_melted,
        x="Model",
        y="R2",
        hue="Dataset",
        palette=["#3498db", "#e74c3c"],
        width=0.6,
        edgecolor="black",
        linewidth=1.2
    )

    # Styling
    plt.title("R² Scores Comparison Across Models", fontsize=20, weight='bold', pad=20)
    plt.xticks(rotation=25, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel("R² Score", fontsize=15, weight='bold')
    plt.xlabel("Model", fontsize=15, weight='bold')
    plt.ylim(0, 1)

    # Add labels on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', fontsize=10)

    plt.legend(title="Dataset", title_fontsize=14, fontsize=13,
               frameon=True, fancybox=True, shadow=True)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # ✅ Save before showing
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved successfully to: {save_path}")

    plt.show()  # Show after saving
    plt.close()

plot()
