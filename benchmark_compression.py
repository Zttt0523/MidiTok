"""
Benchmark tokenization compression: compare token sequence lengths across
tokenization schemes, training models (BPE/Unigram/WordPiece), and vocab sizes.

Finds the tokenization+model combination that achieves maximum compression,
i.e. minimizes the number of tokens a Transformer must process per MIDI file.

Usage:
    python benchmark_compression.py [-o output_dir] [--vocab-sizes 500 1000 2000 5000]
"""

from __future__ import annotations

import argparse
import json
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import miditok
from miditok import TokenizerConfig

MIDI_DIR = Path("tests/MIDIs_one_track")
MIDI_PATHS = sorted(MIDI_DIR.rglob("*.mid"))

TRAINABLE_TOKENIZATIONS = ["REMI", "TSD", "MIDILike", "MMM"]

TRAINING_MODELS = ["BPE", "Unigram", "WordPiece"]

PALETTE = {
    "REMI": "#2196F3",
    "TSD": "#4CAF50",
    "MIDILike": "#F44336",
    "MMM": "#9C27B0",
}

MODEL_MARKERS = {"BPE": "o", "Unigram": "s", "WordPiece": "D"}
MODEL_LINES = {"BPE": "-", "Unigram": "--", "WordPiece": "-."}


def make_tokenizer(tok_name: str) -> miditok.MusicTokenizer:
    """Create an untrained tokenizer instance."""
    cfg = TokenizerConfig(
        num_velocities=16,
        use_tempos=True,
        use_time_signatures=True,
        use_rests=True,
        encode_ids_split="bar",
    )
    if tok_name == "MMM":
        cfg.additional_params["base_tokenizer"] = "TSD"
    return getattr(miditok, tok_name)(cfg)


def get_base_lengths(
    tokenizer: miditok.MusicTokenizer, files: list[Path]
) -> list[int]:
    """Tokenize files without encoding, return base token counts."""
    lengths = []
    for f in files:
        try:
            tokens = tokenizer(f, encode_ids=False)
            if not tokenizer.one_token_stream:
                if isinstance(tokens, list):
                    tokens = tokens[0]
            lengths.append(len(tokens.ids))
        except Exception:
            pass
    return lengths


def train_and_measure(
    tok_name: str,
    model_name: str,
    vocab_size: int,
    files: list[Path],
) -> dict:
    """Train a tokenizer and measure compression."""
    tokenizer = make_tokenizer(tok_name)
    base_lengths = get_base_lengths(tokenizer, files)
    base_vocab_size = len(tokenizer)

    if vocab_size <= base_vocab_size:
        return None

    kwargs = {}
    if model_name == "WordPiece":
        kwargs["max_input_chars_per_word"] = 500

    t0 = time()
    try:
        tokenizer.train(
            vocab_size=vocab_size,
            model=model_name,
            files_paths=files,
            **kwargs,
        )
    except Exception as e:
        return None
    train_time = time() - t0

    encoded_lengths = []
    encode_time = 0
    for f in files:
        try:
            tokens = tokenizer(f, encode_ids=False)
            if not tokenizer.one_token_stream:
                if isinstance(tokens, list):
                    tokens = tokens[0]
            tok_copy = replace(tokens)
            t0 = time()
            tokenizer.encode_token_ids(tok_copy)
            encode_time += time() - t0
            encoded_lengths.append(len(tok_copy.ids))
        except Exception:
            pass

    n = min(len(base_lengths), len(encoded_lengths))
    if n == 0:
        return None

    base_arr = np.array(base_lengths[:n])
    enc_arr = np.array(encoded_lengths[:n])
    ratios = 1 - enc_arr / base_arr

    return {
        "tokenization": tok_name,
        "model": model_name,
        "vocab_size": vocab_size,
        "base_vocab_size": base_vocab_size,
        "final_vocab_size": len(tokenizer),
        "num_files": n,
        "mean_base_len": float(base_arr.mean()),
        "mean_encoded_len": float(enc_arr.mean()),
        "mean_compression_ratio": float(ratios.mean()),
        "median_compression_ratio": float(np.median(ratios)),
        "max_compression_ratio": float(ratios.max()),
        "min_compression_ratio": float(ratios.min()),
        "total_base_tokens": int(base_arr.sum()),
        "total_encoded_tokens": int(enc_arr.sum()),
        "train_time_s": round(train_time, 2),
        "encode_time_s": round(encode_time, 4),
        "base_lengths": base_arr.tolist(),
        "encoded_lengths": enc_arr.tolist(),
    }


def plot_results(results: list[dict], output_dir: Path) -> None:
    """Generate all visualization figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Compression ratio vs vocab size (line plot per tok×model) ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(
        "Token Compression Ratio vs Vocab Size\n"
        "(higher = more compression = better throughput)",
        fontsize=14, fontweight="bold",
    )
    for ax, model_name in zip(axes, TRAINING_MODELS):
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Vocab Size")
        ax.set_ylabel("Mean Compression Ratio" if model_name == "BPE" else "")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.85)

        for tok_name in TRAINABLE_TOKENIZATIONS:
            subset = [
                r for r in results
                if r["tokenization"] == tok_name and r["model"] == model_name
            ]
            if not subset:
                continue
            subset.sort(key=lambda x: x["vocab_size"])
            xs = [r["vocab_size"] for r in subset]
            ys = [r["mean_compression_ratio"] for r in subset]
            ax.plot(
                xs, ys,
                color=PALETTE[tok_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=2, markersize=7,
                label=tok_name,
            )
            for x, y in zip(xs, ys):
                ax.annotate(
                    f"{y:.1%}", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center",
                    color=PALETTE[tok_name],
                )

        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "compression_vs_vocab_size.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Figure 2: Best compression bar chart (at max vocab size) ----
    max_vs = max(r["vocab_size"] for r in results)
    best_results = [r for r in results if r["vocab_size"] == max_vs]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"Compression Ratio Comparison at vocab_size={max_vs}\n"
        "(higher bar = fewer tokens = better model throughput)",
        fontsize=13, fontweight="bold",
    )

    labels = []
    values = []
    colors = []
    hatches = {"BPE": "", "Unigram": "//", "WordPiece": ".."}

    for tok_name in TRAINABLE_TOKENIZATIONS:
        for model_name in TRAINING_MODELS:
            r = next(
                (x for x in best_results
                 if x["tokenization"] == tok_name and x["model"] == model_name),
                None,
            )
            if r:
                labels.append(f"{tok_name}\n{model_name}")
                values.append(r["mean_compression_ratio"])
                colors.append(PALETTE[tok_name])

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Compression Ratio", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # highlight the best
    if values:
        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor("#FFD700")
        bars[best_idx].set_linewidth(3)
        ax.annotate(
            "BEST", (x_pos[best_idx], values[best_idx]),
            textcoords="offset points", xytext=(0, 18),
            fontsize=11, fontweight="bold", color="#FFD700", ha="center",
            arrowprops={"arrowstyle": "->", "color": "#FFD700", "lw": 2},
        )

    fig.tight_layout()
    fig.savefig(output_dir / "compression_bar_chart.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Figure 3: Token length distribution before/after (best combo) ----
    if values:
        best_r = best_results[best_idx]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Token Sequence Length Distribution — {best_r['tokenization']} + "
            f"{best_r['model']} (vocab={best_r['vocab_size']})\n"
            f"Mean compression: {best_r['mean_compression_ratio']:.1%}",
            fontsize=13, fontweight="bold",
        )

        base_lens = best_r["base_lengths"]
        enc_lens = best_r["encoded_lengths"]

        ax = axes[0]
        ax.hist(base_lens, bins=15, color="#B0BEC5", edgecolor="black",
                alpha=0.8, label="Before training")
        ax.hist(enc_lens, bins=15, color=PALETTE[best_r["tokenization"]],
                edgecolor="black", alpha=0.7, label=f"After {best_r['model']}")
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("Number of MIDI files")
        ax.set_title("Length Distribution")
        ax.legend()

        ax = axes[1]
        files = [f.stem[:20] for f in MIDI_PATHS[:len(base_lens)]]
        y_pos = np.arange(len(files))
        ax.barh(y_pos, base_lens, height=0.4, color="#B0BEC5",
                edgecolor="black", linewidth=0.3, label="Base")
        ax.barh(y_pos - 0.4, enc_lens, height=0.4,
                color=PALETTE[best_r["tokenization"]],
                edgecolor="black", linewidth=0.3, label="Encoded")
        ax.set_yticks(y_pos - 0.2)
        ax.set_yticklabels(files, fontsize=7)
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_title("Per-File Comparison")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(output_dir / "length_distribution.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ---- Figure 4: Throughput multiplier heatmap ----
    tok_names = TRAINABLE_TOKENIZATIONS
    model_names = TRAINING_MODELS
    data = np.full((len(tok_names), len(model_names)), np.nan)
    for i, tok in enumerate(tok_names):
        for j, mdl in enumerate(model_names):
            r = next(
                (x for x in best_results
                 if x["tokenization"] == tok and x["model"] == mdl),
                None,
            )
            if r and r["mean_compression_ratio"] > 0:
                data[i, j] = 1 / (1 - r["mean_compression_ratio"])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_yticks(range(len(tok_names)))
    ax.set_yticklabels(tok_names, fontsize=11)

    for i in range(len(tok_names)):
        for j in range(len(model_names)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}x",
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if data[i, j] > np.nanmean(data) else "black")

    ax.set_title(
        f"Throughput Multiplier at vocab_size={max_vs}\n"
        "(= 1/(1-compression_ratio), higher = model processes more music per step)",
        fontsize=12, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Throughput Multiplier")
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table sorted by compression ratio."""
    max_vs = max(r["vocab_size"] for r in results)
    best = sorted(
        [r for r in results if r["vocab_size"] == max_vs],
        key=lambda x: x["mean_compression_ratio"],
        reverse=True,
    )

    header = (
        f"{'Rank':<5} {'Tokenization':<13} {'Model':<10} "
        f"{'Base Vocab':>10} {'Trained Vocab':>13} "
        f"{'Avg Base Len':>13} {'Avg Enc Len':>12} "
        f"{'Compression':>12} {'Throughput':>11} {'Train(s)':>9}"
    )
    print("\n" + "=" * len(header))
    print(f"  COMPRESSION BENCHMARK RESULTS  (vocab_size = {max_vs})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(best, 1):
        throughput = 1 / (1 - r["mean_compression_ratio"]) if r["mean_compression_ratio"] < 1 else float("inf")
        print(
            f"{rank:<5} {r['tokenization']:<13} {r['model']:<10} "
            f"{r['base_vocab_size']:>10} {r['final_vocab_size']:>13} "
            f"{r['mean_base_len']:>13.0f} {r['mean_encoded_len']:>12.0f} "
            f"{r['mean_compression_ratio']:>11.1%} {throughput:>10.2f}x "
            f"{r['train_time_s']:>8.1f}"
        )

    print("=" * len(header))
    if best:
        winner = best[0]
        throughput = 1 / (1 - winner["mean_compression_ratio"])
        print(
            f"\n  WINNER: {winner['tokenization']} + {winner['model']} "
            f"({winner['mean_compression_ratio']:.1%} compression, "
            f"{throughput:.2f}x throughput multiplier)"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization compression for throughput optimization"
    )
    parser.add_argument(
        "-o", "--output-dir", default="benchmark_results",
        help="Output directory for figures and data",
    )
    parser.add_argument(
        "--vocab-sizes", nargs="+", type=int, default=[500, 1000, 2000, 5000],
        help="Vocab sizes to benchmark",
    )
    parser.add_argument(
        "--tokenizations", nargs="+", default=TRAINABLE_TOKENIZATIONS,
        help="Tokenizations to benchmark",
    )
    parser.add_argument(
        "--models", nargs="+", default=TRAINING_MODELS,
        help="Training models to benchmark",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MIDI files: {len(MIDI_PATHS)} from {MIDI_DIR}")
    print(f"Tokenizations: {args.tokenizations}")
    print(f"Models: {args.models}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print()

    all_results = []
    total = len(args.tokenizations) * len(args.models) * len(args.vocab_sizes)
    done = 0

    for tok_name in args.tokenizations:
        for model_name in args.models:
            for vocab_size in args.vocab_sizes:
                done += 1
                tag = f"[{done}/{total}] {tok_name} + {model_name} @ vocab={vocab_size}"
                print(f"{tag} ...", end=" ", flush=True)

                result = train_and_measure(tok_name, model_name, vocab_size, MIDI_PATHS)
                if result:
                    all_results.append(result)
                    print(
                        f"compression={result['mean_compression_ratio']:.1%}  "
                        f"({result['mean_base_len']:.0f} -> "
                        f"{result['mean_encoded_len']:.0f} tokens)  "
                        f"train={result['train_time_s']}s"
                    )
                else:
                    print("SKIPPED")

    # Save raw data
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary_table(all_results)
    plot_results(all_results, output_dir)

    print(f"\nFigures saved to {output_dir}/")
    print("  - compression_vs_vocab_size.png")
    print("  - compression_bar_chart.png")
    print("  - length_distribution.png")
    print("  - throughput_heatmap.png")


if __name__ == "__main__":
    main()
