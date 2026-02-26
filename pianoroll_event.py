"""
Pianoroll-Event: implementation of the encoding scheme from arXiv:2601.19951.

Converts MIDI files into pianoroll-event token sequences and back,
then benchmarks against MidiTok's traditional 1D tokenizations.

Usage:
    python pianoroll_event.py [--frame-len 1] [--block-height 4]
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from symusic import Score, TimeSignature

warnings.filterwarnings("ignore")
import miditok
from miditok import TokenizerConfig

# ---------------------------------------------------------------------------
# Pianoroll-Event Tokenizer
# ---------------------------------------------------------------------------

PITCH_MIN = 21   # A0
PITCH_MAX = 109  # C8
H = PITCH_MAX - PITCH_MIN  # 88
TPQ_TARGET = 16  # 1/16 of a beat, matching the paper


@dataclass
class PianorollEventTokenizer:
    """
    Encode MIDI -> pianoroll -> event tokens following arXiv:2601.19951.

    Parameters:
        frame_len (L): number of timesteps per frame
        block_height (h): pitch rows per event block
        tpq: target ticks-per-quarter for resampling
    """

    frame_len: int = 1
    block_height: int = 4
    tpq: int = TPQ_TARGET

    # built during fit / encoding
    pattern_vocab: dict[bytes, int] = field(default_factory=dict)
    pattern_vocab_inv: dict[int, np.ndarray] = field(default_factory=dict)
    _next_pattern_id: int = 0

    @property
    def K(self) -> int:
        """Number of blocks per frame along the pitch axis."""
        return int(np.ceil(H / self.block_height))

    @property
    def vocab_size(self) -> int:
        n_frame = self.K           # Frame(0) .. Frame(K-1)
        n_gap = self.K - 1         # Gap(1) .. Gap(K-1)
        n_pattern = len(self.pattern_vocab)
        n_musical = 4              # Bar, TimeSig(num, den), EndOfTrack, Pad
        return n_frame + n_gap + n_pattern + n_musical

    # ---- encoding ----

    def _score_to_pianoroll(self, score: Score) -> tuple[np.ndarray, list]:
        """Return (H, T) binary pianoroll and list of bar/time-sig events."""
        score = score.resample(tpq=self.tpq)
        combined = np.zeros((H, 0), dtype=np.uint8)
        for track in score.tracks:
            pr = track.pianoroll(modes=["frame"], pitch_range=(PITCH_MIN, PITCH_MAX))
            pr = pr[0]  # (H, T)
            if pr.shape[1] > combined.shape[1]:
                combined = np.pad(
                    combined,
                    ((0, 0), (0, pr.shape[1] - combined.shape[1])),
                )
            combined[:, : pr.shape[1]] = np.maximum(
                combined[:, : pr.shape[1]], (pr > 0).astype(np.uint8)
            )

        # musical structure: bar positions
        bar_events = []
        beat_ticks = self.tpq  # ticks per beat (quarter note)
        for ts in score.time_signatures:
            bar_events.append(
                ("TimeSig", ts.time, ts.numerator, ts.denominator)
            )

        return combined, bar_events

    def _get_pattern_id(self, block: np.ndarray) -> int:
        """Map a block to a pattern token id, adding to vocab if new."""
        key = block.tobytes()
        if key not in self.pattern_vocab:
            pid = self._next_pattern_id
            self.pattern_vocab[key] = pid
            self.pattern_vocab_inv[pid] = block.copy()
            self._next_pattern_id += 1
        return self.pattern_vocab[key]

    def _encode_frame(self, frame: np.ndarray) -> list[str]:
        """Encode a single frame (H, L) into event tokens (Algorithm 1)."""
        h = self.block_height
        K = self.K
        blocks = []
        for j in range(K):
            start = j * h
            end = min((j + 1) * h, H)
            blk = frame[start:end, :]
            if blk.shape[0] < h:
                blk = np.pad(blk, ((0, h - blk.shape[0]), (0, 0)))
            blocks.append(blk)

        # Find leading empty blocks -> Frame Event
        leading_empty = 0
        for blk in blocks:
            if blk.any():
                break
            leading_empty += 1

        if leading_empty == K:
            return [f"Frame_{leading_empty}"]

        # Find trailing empty blocks (will be dropped)
        trailing_empty = 0
        for blk in reversed(blocks):
            if blk.any():
                break
            trailing_empty += 1

        events = [f"Frame_{leading_empty}"]
        gap_count = 0
        end_idx = K - trailing_empty

        for j in range(leading_empty, end_idx):
            blk = blocks[j]
            if not blk.any():
                gap_count += 1
            else:
                if gap_count > 0:
                    events.append(f"Gap_{gap_count}")
                    gap_count = 0
                pid = self._get_pattern_id(blk)
                events.append(f"Pattern_{pid}")

        if gap_count > 0:
            events.append(f"Gap_{gap_count}")

        return events

    def encode(self, score_or_path) -> tuple[list[str], dict]:
        """
        Encode a Score or MIDI path into Pianoroll-Event token sequence.

        Returns (tokens, info_dict).
        """
        if isinstance(score_or_path, (str, Path)):
            score = Score(str(score_or_path))
        else:
            score = score_or_path

        pianoroll, bar_events = self._score_to_pianoroll(score)
        T = pianoroll.shape[1]
        L = self.frame_len
        N = int(np.ceil(T / L))

        # pre-compute bar ticks
        tpq = self.tpq
        bar_ticks = set()
        ts_at_tick = {}
        current_ts = (4, 4)
        for ev in bar_events:
            if ev[0] == "TimeSig":
                ts_at_tick[ev[1]] = (ev[2], ev[3])

        # generate bar tick positions
        tick = 0
        while tick <= T:
            if tick in ts_at_tick:
                current_ts = ts_at_tick[tick]
            bar_ticks.add(tick)
            bar_len = current_ts[0] * tpq * 4 // current_ts[1]
            tick += bar_len

        tokens = []
        current_ts_str = None

        for i in range(N):
            t_start = i * L
            t_end = min((i + 1) * L, T)
            frame = pianoroll[:, t_start:t_end]
            if frame.shape[1] < L:
                frame = np.pad(frame, ((0, 0), (0, L - frame.shape[1])))

            # Insert Musical Structure Events at bar boundaries
            for t in range(t_start, t_end):
                if t in bar_ticks:
                    tokens.append("Bar")
                if t in ts_at_tick:
                    num, den = ts_at_tick[t]
                    ts_str = f"TimeSig_{num}/{den}"
                    if ts_str != current_ts_str:
                        tokens.append(ts_str)
                        current_ts_str = ts_str

            frame_events = self._encode_frame(frame)
            tokens.extend(frame_events)

        info = {
            "pianoroll_shape": pianoroll.shape,
            "num_frames": N,
            "num_tokens": len(tokens),
            "vocab_size": self.vocab_size,
            "num_patterns": len(self.pattern_vocab),
            "sparsity": 1 - np.count_nonzero(pianoroll) / pianoroll.size,
        }
        return tokens, info

    def decode(self, tokens: list[str]) -> np.ndarray:
        """Decode tokens back into pianoroll (H, T_approx)."""
        h = self.block_height
        K = self.K
        L = self.frame_len

        frames = []
        current_frame = np.zeros((H, L), dtype=np.uint8)
        block_cursor = 0
        in_frame = False

        for tok in tokens:
            if tok.startswith("Frame_"):
                if in_frame:
                    frames.append(current_frame.copy())
                current_frame = np.zeros((H, L), dtype=np.uint8)
                block_cursor = int(tok.split("_")[1])
                in_frame = True

            elif tok.startswith("Gap_"):
                block_cursor += int(tok.split("_")[1])

            elif tok.startswith("Pattern_"):
                pid = int(tok.split("_")[1])
                if pid in self.pattern_vocab_inv:
                    blk = self.pattern_vocab_inv[pid]
                    start = block_cursor * h
                    end = min(start + h, H)
                    rows = end - start
                    current_frame[start:end, :] = blk[:rows, :]
                block_cursor += 1

            # Bar, TimeSig, etc. are skipped during decode

        if in_frame:
            frames.append(current_frame)

        if not frames:
            return np.zeros((H, 0), dtype=np.uint8)

        return np.concatenate(frames, axis=1)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def compute_bdi(avg_seq_len: float, vocab_size: int) -> float:
    """Budget-Aware Difficulty Index (Eq. 3 from paper)."""
    return avg_seq_len ** 2 * np.sqrt(vocab_size)


def benchmark_miditok(
    tok_name: str, files: list[Path]
) -> dict:
    """Tokenize files with a MidiTok tokenizer and return stats."""
    cfg = TokenizerConfig(num_velocities=16)
    if tok_name == "MMM":
        cfg.additional_params["base_tokenizer"] = "TSD"
    try:
        tokenizer = getattr(miditok, tok_name)(cfg)
    except Exception:
        return None

    lengths = []
    for f in files:
        try:
            tokens = tokenizer(f)
            if isinstance(tokens, list):
                total = sum(len(t.ids) for t in tokens)
            else:
                total = len(tokens.ids)
            lengths.append(total)
        except Exception:
            pass

    if not lengths:
        return None

    return {
        "method": tok_name,
        "vocab_size": len(tokenizer),
        "avg_seq_len": np.mean(lengths),
        "median_seq_len": np.median(lengths),
        "total_tokens": sum(lengths),
        "num_files": len(lengths),
        "lengths": lengths,
    }


def benchmark_pianoroll_event(
    files: list[Path],
    frame_len: int = 1,
    block_height: int = 4,
) -> dict:
    """Tokenize files with Pianoroll-Event and return stats."""
    tokenizer = PianorollEventTokenizer(
        frame_len=frame_len, block_height=block_height
    )

    lengths = []
    all_info = []
    for f in files:
        try:
            tokens, info = tokenizer.encode(f)
            lengths.append(len(tokens))
            all_info.append(info)
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")

    if not lengths:
        return None

    return {
        "method": f"PR-Event(L={frame_len},h={block_height})",
        "vocab_size": tokenizer.vocab_size,
        "num_patterns": len(tokenizer.pattern_vocab),
        "avg_seq_len": np.mean(lengths),
        "median_seq_len": np.median(lengths),
        "total_tokens": sum(lengths),
        "num_files": len(lengths),
        "lengths": lengths,
        "avg_sparsity": np.mean([i["sparsity"] for i in all_info]),
    }


def verify_roundtrip(file_path: Path, frame_len: int, block_height: int) -> dict:
    """Verify encode-decode round-trip accuracy."""
    tokenizer = PianorollEventTokenizer(
        frame_len=frame_len, block_height=block_height
    )
    score = Score(str(file_path))
    pianoroll_orig, _ = tokenizer._score_to_pianoroll(score)

    tokens, info = tokenizer.encode(score)
    pianoroll_decoded = tokenizer.decode(tokens)

    # Trim to common length
    T = min(pianoroll_orig.shape[1], pianoroll_decoded.shape[1])
    orig = pianoroll_orig[:, :T]
    decoded = pianoroll_decoded[:, :T]

    match = (orig == decoded)
    accuracy = match.sum() / match.size
    note_recall = (
        decoded[orig > 0].sum() / max(orig.sum(), 1)
    )
    note_precision = (
        orig[decoded > 0].sum() / max(decoded.sum(), 1)
    )

    return {
        "file": file_path.name,
        "orig_shape": pianoroll_orig.shape,
        "decoded_shape": pianoroll_decoded.shape,
        "accuracy": float(accuracy),
        "note_recall": float(note_recall),
        "note_precision": float(note_precision),
        "num_tokens": len(tokens),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(results: list[dict], output_path: Path) -> None:
    """Generate comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Pianoroll-Event vs MidiTok Tokenizations\n"
        "(encoding efficiency comparison on MidiTok test corpus)",
        fontsize=14, fontweight="bold",
    )

    methods = [r["method"] for r in results]
    colors = []
    palette = {
        "REMI": "#2196F3", "TSD": "#4CAF50", "MIDILike": "#F44336",
        "MMM": "#9C27B0", "Octuple": "#FF9800", "CPWord": "#00BCD4",
        "MuMIDI": "#795548",
    }
    for m in methods:
        if m.startswith("PR-Event"):
            colors.append("#E91E63")
        else:
            colors.append(palette.get(m, "#607D8B"))

    # 1) Average sequence length
    ax = axes[0]
    vals = [r["avg_seq_len"] for r in results]
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Sequence Length")
    ax.set_title("Sequence Length (lower = better)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # 2) Vocab size
    ax = axes[1]
    vals = [r["vocab_size"] for r in results]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Vocabulary Size")
    ax.set_title("Vocabulary Size")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val}", ha="center", va="bottom", fontsize=7)

    # 3) BDI
    ax = axes[2]
    vals = [compute_bdi(r["avg_seq_len"], r["vocab_size"]) for r in results]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("BDI (l² × √V)")
    ax.set_title("Budget-Aware Difficulty Index (lower = better)")
    ax.set_yscale("log")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=6)

    # Highlight best
    best_idx = int(np.argmin(vals))
    bars[best_idx].set_edgecolor("#FFD700")
    bars[best_idx].set_linewidth(3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_hyperparameter_sweep(sweep_results: list[dict], output_path: Path) -> None:
    """Plot how L and h affect sequence length, vocab, and BDI."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Pianoroll-Event Hyperparameter Sensitivity\n"
        "(frame_len L × block_height h)",
        fontsize=13, fontweight="bold",
    )

    # Group by block_height
    h_values = sorted(set(r["h"] for r in sweep_results))
    h_colors = plt.cm.Set1(np.linspace(0, 0.8, len(h_values)))

    for idx, (ax, metric, label, log) in enumerate([
        (axes[0], "avg_seq_len", "Avg Sequence Length", False),
        (axes[1], "vocab_size", "Vocabulary Size", True),
        (axes[2], "bdi", "BDI (l² × √V)", True),
    ]):
        for hi, color in zip(h_values, h_colors):
            subset = sorted(
                [r for r in sweep_results if r["h"] == hi],
                key=lambda r: r["L"],
            )
            xs = [r["L"] for r in subset]
            ys = [r[metric] for r in subset]
            ax.plot(xs, ys, "o-", color=color, label=f"h={hi}", linewidth=2,
                    markersize=6)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}" if not log else f"{y:.1e}",
                            (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=6, color=color)

        ax.set_xlabel("Frame Length (L)")
        ax.set_ylabel(label)
        ax.set_title(label)
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pianoroll-Event analysis & comparison"
    )
    parser.add_argument("-o", "--output-dir", default="pianoroll_event_results")
    parser.add_argument("--frame-len", type=int, default=1)
    parser.add_argument("--block-height", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_dir = Path("tests/MIDIs_one_track")
    files = sorted(midi_dir.rglob("*.mid"))
    files = [f for f in files if f.name != "empty.mid"]
    print(f"Test corpus: {len(files)} MIDI files from {midi_dir}\n")

    # ---- Step 1: Round-trip verification ----
    print("=" * 70)
    print("STEP 1: ROUND-TRIP VERIFICATION")
    print("=" * 70)
    for f in files[:3]:
        rt = verify_roundtrip(f, args.frame_len, args.block_height)
        print(
            f"  {rt['file']:40s}  accuracy={rt['accuracy']:.6f}  "
            f"recall={rt['note_recall']:.4f}  precision={rt['note_precision']:.4f}  "
            f"tokens={rt['num_tokens']}"
        )
    print()

    # ---- Step 2: Hyperparameter sweep ----
    print("=" * 70)
    print("STEP 2: HYPERPARAMETER SWEEP (L × h)")
    print("=" * 70)
    sweep_results = []
    for L in [1, 2, 4]:
        for h in [1, 2, 4, 8, 11]:
            result = benchmark_pianoroll_event(files, frame_len=L, block_height=h)
            if result:
                bdi = compute_bdi(result["avg_seq_len"], result["vocab_size"])
                sweep_results.append({
                    "L": L, "h": h,
                    "avg_seq_len": result["avg_seq_len"],
                    "vocab_size": result["vocab_size"],
                    "num_patterns": result["num_patterns"],
                    "bdi": bdi,
                })
                print(
                    f"  L={L}, h={h:2d}  →  "
                    f"seq_len={result['avg_seq_len']:8.0f}  "
                    f"vocab={result['vocab_size']:6d}  "
                    f"patterns={result['num_patterns']:5d}  "
                    f"BDI={bdi:.2e}"
                )
    print()

    plot_hyperparameter_sweep(sweep_results, output_dir / "hyperparameter_sweep.png")

    # Find best hyperparams
    best_sweep = min(sweep_results, key=lambda r: r["bdi"])
    best_L, best_h = best_sweep["L"], best_sweep["h"]
    print(f"  Best (L={best_L}, h={best_h}): "
          f"BDI={best_sweep['bdi']:.2e}, "
          f"seq_len={best_sweep['avg_seq_len']:.0f}, "
          f"vocab={best_sweep['vocab_size']}")
    print()

    # ---- Step 3: Full comparison with MidiTok ----
    print("=" * 70)
    print("STEP 3: COMPARISON WITH MIDITOK TOKENIZATIONS")
    print("=" * 70)
    all_results = []

    # Pianoroll-Event with best params
    pr_result = benchmark_pianoroll_event(files, frame_len=best_L, block_height=best_h)
    if pr_result:
        all_results.append(pr_result)

    # Also add the paper's claimed config (L=1, h=4)
    if (best_L, best_h) != (1, 4):
        pr_paper = benchmark_pianoroll_event(files, frame_len=1, block_height=4)
        if pr_paper:
            pr_paper["method"] = "PR-Event(paper)"
            all_results.append(pr_paper)

    # MidiTok tokenizations (no velocity for fair comparison per paper)
    for tok_name in ["REMI", "TSD", "MIDILike", "CPWord", "Octuple", "MuMIDI", "MMM"]:
        result = benchmark_miditok(tok_name, files)
        if result:
            all_results.append(result)

    # Print comparison table
    header = (
        f"{'Method':<25} {'Vocab':>6} {'Avg Len':>10} "
        f"{'BDI':>12} {'vs Best':>8}"
    )
    print(header)
    print("-" * len(header))

    bdis = []
    for r in all_results:
        bdi = compute_bdi(r["avg_seq_len"], r["vocab_size"])
        bdis.append(bdi)

    min_bdi = min(bdis)
    sorted_idx = np.argsort(bdis)

    for idx in sorted_idx:
        r = all_results[idx]
        bdi = bdis[idx]
        ratio = bdi / min_bdi
        marker = " ★" if idx == sorted_idx[0] else ""
        print(
            f"{r['method']:<25} {r['vocab_size']:>6} "
            f"{r['avg_seq_len']:>10.0f} "
            f"{bdi:>12.2e} {ratio:>7.2f}×{marker}"
        )

    print()

    # Save results
    serializable = []
    for r, bdi in zip(all_results, bdis):
        entry = {k: v for k, v in r.items() if k != "lengths"}
        entry["bdi"] = bdi
        serializable.append(entry)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    plot_comparison(all_results, output_dir / "comparison.png")

    # ---- Step 4: Analysis summary ----
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    pr_main = all_results[0]
    pr_bdi = bdis[0]

    remi_r = next((r for r in all_results if r["method"] == "REMI"), None)
    midilike_r = next((r for r in all_results if r["method"] == "MIDILike"), None)

    if remi_r:
        remi_bdi = compute_bdi(remi_r["avg_seq_len"], remi_r["vocab_size"])
        print(f"  vs REMI:     {remi_bdi/pr_bdi:.2f}× BDI improvement  "
              f"(seq: {remi_r['avg_seq_len']:.0f} vs {pr_main['avg_seq_len']:.0f})")
    if midilike_r:
        ml_bdi = compute_bdi(midilike_r["avg_seq_len"], midilike_r["vocab_size"])
        print(f"  vs MIDILike: {ml_bdi/pr_bdi:.2f}× BDI improvement  "
              f"(seq: {midilike_r['avg_seq_len']:.0f} vs {pr_main['avg_seq_len']:.0f})")

    print(f"\n  Pianoroll-Event vocab breakdown:")
    print(f"    Pattern tokens: {pr_main.get('num_patterns', 'N/A')}")
    print(f"    Total vocab:    {pr_main['vocab_size']}")

    print(f"\n  Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
