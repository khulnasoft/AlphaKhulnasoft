"""Visualizations for proof-generation benchmarks.

Mirrors ``alphakhulnasoft.visualizer.AlphaPlotter`` but plots proof-specific
metrics (validity, iterations, proof depth) from a proof benchmark output
file produced by ``proof_benchmark.run_proof_benchmark``.
"""

import json
import os

# Fix matplotlib backend for Colab/headless environments
os.environ.pop("MPLBACKEND", None)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ProofPlotter:
    """Generates research-grade visualizations for proof benchmarks."""

    def __init__(self, results_file: str):
        with open(results_file) as f:
            payload = json.load(f)
        self.records = payload.get("results", payload if isinstance(payload, list) else [])
        self.df = pd.DataFrame(self.records)

        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except Exception:
            plt.style.use("ggplot")
        sns.set_context("talk")

    def plot_proof_trajectory(self, output_path="proof_curve.png"):
        """Visualizes WHEN theorems were proven (iteration 1 vs later)."""
        if self.df.empty:
            print("⚠️ No data to plot.")
            return

        proven = self.df[self.df["is_valid"]].copy()
        if proven.empty:
            print("⚠️ No proven theorems to plot.")
            return

        max_iter = self.df["iterations"].max()
        all_iters = range(1, max_iter + 1)
        iter_counts = (
            proven["iterations"].value_counts().reindex(all_iters, fill_value=0).sort_index()
        )
        cumulative = iter_counts.cumsum()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=iter_counts.index,
            y=iter_counts.values,
            color="skyblue",
            label="Proven at Iteration X",
        )
        sns.lineplot(
            x=range(len(iter_counts)),
            y=cumulative.values,
            marker="o",
            color="crimson",
            linewidth=3,
            label="Cumulative Proven",
        )
        plt.title(f"Proof Repair Trajectory (N={len(self.df)})", fontsize=16, fontweight="bold")
        plt.xlabel("Iteration Number")
        plt.ylabel("Theorems Proven")
        plt.xticks(ticks=range(len(iter_counts)), labels=iter_counts.index)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"📊 Saved Proof Trajectory to {output_path}")

    def plot_proof_depth_matrix(self, output_path="proof_depth_matrix.png"):
        """Scatter plot of proof depth vs iterations, colored by validity."""
        if self.df.empty:
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df,
            x="depth",
            y="iterations",
            hue="is_valid",
            style="is_valid",
            s=200,
            palette={True: "green", False: "red"},
        )
        plt.title("Proof Depth vs. Repair Iterations", fontsize=16)
        plt.xlabel("Proof Depth (nesting)")
        plt.ylabel("Iterations Required")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"📊 Saved Proof Depth Matrix to {output_path}")

    def generate_report(self):
        """Prints a text summary of the proof metrics."""
        total = len(self.df)
        proven = self.df["is_valid"].sum() if "is_valid" in self.df else 0
        proof_rate = (proven / total) * 100 if total else 0
        avg_iters = self.df[self.df["is_valid"]]["iterations"].mean() if proven > 0 else 0

        print("\n" + "=" * 30)
        print("📈 ALPHA PROOF REPORT")
        print("=" * 30)
        print(f"Total Theorems: {total}")
        print(f"Proven:         {proven} ({proof_rate:.1f}%)")
        print(f"Avg Iters:      {avg_iters:.2f}")
        print("=" * 30 + "\n")

    @staticmethod
    def read_summary(results_file: str) -> dict:
        with open(results_file) as f:
            payload = json.load(f)
        summary: dict = payload.get("summary", {})
        return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        plotter = ProofPlotter(sys.argv[1])
        plotter.generate_report()
        plotter.plot_proof_trajectory()
        plotter.plot_proof_depth_matrix()
    else:
        print("Usage: python -m alphakhulnasoft.proof_visualizer <results_file.json>")
