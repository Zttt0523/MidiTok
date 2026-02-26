"""
Symbolic Music Generation: Bottleneck Analysis Visualization.

Creates a diagnostic figure mapping the key bottlenecks
based on empirical findings from our MidiTok / Pianoroll-Event experiments
and current literature survey.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/opt/cursor/artifacts/bottleneck_analysis.png"


def main():
    fig = plt.figure(figsize=(20, 14), facecolor="white")

    # ---- Main bottleneck ranking ----
    ax1 = fig.add_axes([0.05, 0.42, 0.55, 0.52])

    bottlenecks = [
        ("Long-range\nStructure & Form", 9.2,
         "Models lose coherence after ~30s;\nhierarchical planning is nascent"),
        ("Evaluation\nMethodology", 8.5,
         "Objective metrics ≠ human perception;\nno standardized benchmark"),
        ("Data Quality\n& Completeness", 7.8,
         "MIDI lacks expression/timbre;\nmost datasets < 50K pieces"),
        ("Sequence Length\nScaling", 7.0,
         "5-min piece = 10K-50K tokens;\nO(n²) attention is prohibitive"),
        ("Tokenization\n& Representation", 5.5,
         "Mature solutions exist (REMI/TSD+BPE);\n2D methods promising but marginal gains"),
        ("Model\nArchitecture", 4.8,
         "Transformers work well;\nrelative attention helps but not decisive"),
        ("Controllability\n& Editing", 6.5,
         "Style/structure control still crude;\ninteractive editing is rare"),
    ]

    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    labels = [b[0] for b in bottlenecks]
    scores = [b[1] for b in bottlenecks]
    descs = [b[2] for b in bottlenecks]

    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(bottlenecks)))

    y = np.arange(len(labels))
    bars = ax1.barh(y, scores, color=colors, edgecolor="black", linewidth=0.5,
                    height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Bottleneck Severity (1-10)", fontsize=11)
    ax1.set_xlim(0, 11)
    ax1.invert_yaxis()
    ax1.set_title(
        "Symbolic Music Generation: Bottleneck Severity Ranking\n"
        "(based on empirical analysis & 2025-2026 literature)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, axis="x", alpha=0.3)

    for bar, score, desc in zip(bars, scores, descs):
        ax1.text(score + 0.15, bar.get_y() + bar.get_height() / 2,
                 f"{score}/10  {desc}", va="center", fontsize=7.5,
                 style="italic", color="#333")

    # ---- Maturity vs Impact scatter ----
    ax2 = fig.add_axes([0.65, 0.42, 0.32, 0.52])

    areas = {
        "Tokenization": (8.0, 5.5, "#4CAF50"),
        "Architecture": (7.5, 4.8, "#2196F3"),
        "Long-range\nStructure": (3.0, 9.2, "#F44336"),
        "Evaluation": (2.5, 8.5, "#FF9800"),
        "Data": (4.5, 7.8, "#9C27B0"),
        "Seq Length": (5.0, 7.0, "#00BCD4"),
        "Control": (3.5, 6.5, "#795548"),
    }

    for label, (maturity, impact, color) in areas.items():
        ax2.scatter(maturity, impact, s=300, c=color, edgecolors="black",
                    linewidth=1, zorder=3)
        ax2.annotate(label, (maturity, impact),
                     textcoords="offset points", xytext=(12, 0),
                     fontsize=8, fontweight="bold", color=color)

    ax2.set_xlabel("Solution Maturity (1-10)", fontsize=10)
    ax2.set_ylabel("Bottleneck Severity (1-10)", fontsize=10)
    ax2.set_title("Maturity vs Severity\n(top-left = biggest opportunities)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlim(1, 10)
    ax2.set_ylim(3, 10.5)
    ax2.grid(True, alpha=0.3)

    # Quadrant labels
    ax2.axhline(y=6.5, color="gray", linestyle="--", alpha=0.3)
    ax2.axvline(x=5.5, color="gray", linestyle="--", alpha=0.3)
    ax2.text(2, 10.2, "HIGH PRIORITY\n(severe + immature)", fontsize=7,
             color="red", alpha=0.5, ha="center")
    ax2.text(8.5, 10.2, "Monitor\n(severe but mature)", fontsize=7,
             color="orange", alpha=0.5, ha="center")
    ax2.text(2, 3.5, "Explore\n(mild + immature)", fontsize=7,
             color="blue", alpha=0.4, ha="center")
    ax2.text(8.5, 3.5, "Solved\n(mild + mature)", fontsize=7,
             color="green", alpha=0.4, ha="center")

    # ---- Bottom panel: evidence from our experiments ----
    ax3 = fig.add_axes([0.05, 0.03, 0.90, 0.32])
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    evidence_text = """EMPIRICAL EVIDENCE FROM OUR EXPERIMENTS

+-- Tokenization (Maturity: 8/10) -----------------------------------------------------------------------+
|  * MidiTok REMI/TSD are mature; BPE achieves 58.6% compression (2.42x throughput gain)                  |
|  * Pianoroll-Event's 2D encoding does NOT beat 1D methods on BDI, but structural preservation may help  |
|  * Conclusion: tokenization is NOT the main bottleneck -- existing solutions are good enough             |
+---------------------------------------------------------------------------------------------------------+

+-- The Real Bottleneck: Long-Range Structure (Maturity: 3/10) -------------------------------------------+
|  * Our MIDI files (57-900s) produce 10K-50K tokens, far beyond effective Transformer attention range     |
|  * Pianoroll-Event paper only tests 40-90s short pieces, avoiding the long-range structure problem       |
|  * Latest work (YuE, MusicWeaver, BACH) all attempt hierarchical planning, but no consensus yet         |
+---------------------------------------------------------------------------------------------------------+

+-- The Underestimated Bottleneck: Evaluation (Maturity: 2.5/10) -----------------------------------------+
|  * Pianoroll-Event uses PR/GC/SC + MOS -- objective vs subjective metrics are misaligned (confirmed)     |
|  * No reliable automatic evaluation metric = can't iterate fast = entire field's progress is blocked     |
|  * Analogy: NLP had BLEU/ROUGE -- accelerated MT for a decade; music lacks an equivalent                |
+---------------------------------------------------------------------------------------------------------+"""

    ax3.text(0.0, 1.0, evidence_text, transform=ax3.transAxes,
             fontsize=8, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD", alpha=0.9))

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
