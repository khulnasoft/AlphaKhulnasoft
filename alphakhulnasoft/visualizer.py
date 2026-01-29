import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AlphaPlotter:
    """
    Generates research-grade visualizations for the AlphaKhulnasoft Benchmark.
    """

    def __init__(self, results_file: str):
        with open(results_file) as f:
            self.data = json.load(f)
        self.df = pd.DataFrame(self.data)

        # Basic styling
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except Exception:
            plt.style.use("ggplot")  # Fallback if seaborn style is missing
        sns.set_context("talk")  # Bigger fonts for presentation

    def plot_repair_trajectory(self, output_path="repair_curve.png"):
        """
        Visualizes WHEN problems were solved (Iter 1 vs Iter 5).
        This proves the value of the 'Repair Loop'.
        """
        if self.df.empty:
            print("⚠️ No data to plot.")
            return

        # Filter only solved problems
        solved = self.df[self.df["pass"]].copy()

        if solved.empty:
            print("⚠️ No solved problems to plot.")
            return

        # Count how many solved at each iteration count
        # Ensure we have all iterations represented from 1 to max found or at least 1-5
        max_iter = self.df["iterations"].max()
        all_iters = range(1, max_iter + 1)

        iter_counts = (
            solved["iterations"].value_counts().reindex(all_iters, fill_value=0).sort_index()
        )

        # Cumulative Sum (Pass@K equivalent)
        cumulative = iter_counts.cumsum()

        plt.figure(figsize=(10, 6))

        # Bar chart for specific iteration wins
        sns.barplot(
            x=iter_counts.index,
            y=iter_counts.values,
            color="skyblue",
            label="Solved at Iteration X",
        )

        # Line chart for cumulative success
        sns.lineplot(
            x=range(len(iter_counts)),
            y=cumulative.values,
            marker="o",
            color="crimson",
            linewidth=3,
            label="Cumulative Solved (Pass@K)",
        )

        plt.title(
            f"AlphaKhulnasoft Repair Trajectory (N={len(self.df)})", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Iteration Number")
        plt.ylabel("Problems Solved")
        plt.xticks(ticks=range(len(iter_counts)), labels=iter_counts.index)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_path)
        print(f"📊 Saved Repair Curve to {output_path}")

    def plot_efficiency_matrix(self, output_path="efficiency_matrix.png"):
        """
        Scatter plot of Duration vs Iterations.
        Good agents stay in the bottom-left (Fast + Few Iters).
        """
        if self.df.empty:
            return

        plt.figure(figsize=(10, 6))

        # Color by Solved/Failed
        sns.scatterplot(
            data=self.df,
            x="duration",
            y="iterations",
            hue="pass",
            style="pass",
            s=200,
            palette={True: "green", False: "red"},
        )

        plt.title("Efficiency Matrix: Time vs. Retries", fontsize=16)
        plt.xlabel("Execution Time (s)")
        plt.ylabel("Iterations Required")

        # Add a "Sweet Spot" box
        max_duration = self.df["duration"].max() * 1.1 if not self.df.empty else 10
        plt.xlim(0, max_duration)
        plt.ylim(0, self.df["iterations"].max() + 1)

        plt.axvspan(0, 5, ymin=0, ymax=0.3, color="green", alpha=0.1, label="High Efficiency Zone")

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"📊 Saved Efficiency Matrix to {output_path}")

    def generate_report(self):
        """Prints a text summary of the metrics."""
        total = len(self.df)
        solved = self.df["pass"].sum()
        pass_rate = (solved / total) * 100
        avg_iters = self.df[self.df["pass"]]["iterations"].mean() if solved > 0 else 0

        print("\n" + "=" * 30)
        print("📈 ALPHA PERFORMANCE REPORT")
        print("=" * 30)
        print(f"Total Problems:  {total}")
        print(f"Solved:          {solved} ({pass_rate:.1f}%)")
        print(f"Avg Iters (Fix): {avg_iters:.2f}")
        print("=" * 30 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        plotter = AlphaPlotter(sys.argv[1])
        plotter.generate_report()
        plotter.plot_repair_trajectory()
        plotter.plot_efficiency_matrix()
    else:
        print("Usage: python -m alphakhulnasoft.visualizer <results_file.json>")
