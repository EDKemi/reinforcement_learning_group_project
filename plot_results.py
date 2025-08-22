import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/eval_log.csv")

    steps = df["step"]
    avg = df["avg_return"]
    std = df["std_return"]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, avg, label="Average Return", color="blue")
    plt.fill_between(steps, avg - std, avg + std, color="blue", alpha=0.2, label="Std Dev")

    plt.title("SAC Training Performance on BipedalWalker")
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/performance_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
