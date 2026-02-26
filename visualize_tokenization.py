"""
Visualize MIDI tokenization: compare notes before and after tokenize->detokenize.

Produces a figure with:
  1. Top row: side-by-side piano-roll comparison (original vs. reconstructed)
  2. Middle row: overlay diff view - matching notes in green, deviations in red
  3. Bottom row: token stream annotation strip (first N tokens)

Usage:
    python visualize_tokenization.py [midi_path] [--tokenization REMI]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from symusic import Score

import miditok
from miditok import TokenizerConfig


def notes_to_arrays(track):
    """Return (starts, ends, pitches, velocities) as numpy arrays."""
    n = track.notes.numpy()
    return n["time"], n["time"] + n["duration"], n["pitch"], n["velocity"]


def draw_piano_roll(ax, starts, ends, pitches, velocities, color_fn, alpha=0.85,
                    label=None):
    """Draw a piano-roll on *ax* using colored rectangles."""
    if len(starts) == 0:
        return
    rects = []
    colors = []
    for s, e, p, v in zip(starts, ends, pitches, velocities):
        w = max(e - s, 0.3)
        rect = mpatches.FancyBboxPatch(
            (s, p - 0.4), w, 0.8, boxstyle="round,pad=0.02",
        )
        rects.append(rect)
        colors.append(color_fn(v))
    pc = PatchCollection(rects, facecolors=colors, edgecolors="none", alpha=alpha)
    ax.add_collection(pc)
    if label:
        ax.set_title(label, fontsize=11, fontweight="bold")


def setup_pianoroll_ax(ax, starts, ends, pitches, margin=2):
    """Configure axis limits and styling for a piano-roll plot."""
    if len(starts) == 0:
        return
    t_min = max(0, int(starts.min()) - margin)
    t_max = int(ends.max()) + margin
    p_min = int(pitches.min()) - 2
    p_max = int(pitches.max()) + 2
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(p_min, p_max)
    ax.set_xlabel("Ticks", fontsize=9)
    ax.set_ylabel("MIDI Pitch", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.5)


TOKEN_TYPE_COLORS = {
    "Bar": "#2196F3", "Position": "#4CAF50", "Pitch": "#F44336",
    "Velocity": "#FF9800", "Duration": "#9C27B0", "Program": "#00BCD4",
    "Tempo": "#795548", "TimeSig": "#607D8B", "Chord": "#E91E63",
    "Rest": "#CDDC39", "NoteOn": "#F44336", "NoteOff": "#3F51B5",
    "TimeShift": "#4CAF50",
}


def draw_token_strip(ax, tokens, max_tokens=200):
    """Draw a horizontal strip of colored token blocks."""
    n = min(len(tokens), max_tokens)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    for i in range(n):
        tok_type = tokens[i].split("_")[0]
        color = TOKEN_TYPE_COLORS.get(tok_type, "#BDBDBD")
        ax.barh(0.5, 1, left=i, height=0.8, color=color, edgecolor="white",
                linewidth=0.3, align="center")
        if n <= 80:
            ax.text(i + 0.5, 0.5, tok_type[:3], ha="center", va="center",
                    fontsize=5, rotation=90, color="white", fontweight="bold")
    seen = set()
    legend_types = [t.split("_")[0] for t in tokens[:n]
                    if not (t.split("_")[0] in seen or seen.add(t.split("_")[0]))]
    handles = [mpatches.Patch(color=TOKEN_TYPE_COLORS.get(t, "#BDBDBD"), label=t)
               for t in legend_types]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              ncol=min(len(handles), 6), framealpha=0.8)
    ax.set_xlabel(f"Token index (showing first {n} of {len(tokens)})", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Token stream (color = token type)", fontsize=10, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Visualize MIDI tokenization")
    parser.add_argument("midi", nargs="?",
                        default="tests/MIDIs_one_track/Maestro_1.mid")
    parser.add_argument("--tokenization", default="REMI",
                        choices=miditok.tokenizations.__all__)
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Show only first N ticks (0 = all)")
    parser.add_argument("-o", "--output", default="tokenization_comparison.png")
    args = parser.parse_args()

    midi_path = Path(args.midi)
    print(f"Loading {midi_path} ...")
    original = Score(str(midi_path))

    config = TokenizerConfig(num_velocities=16, use_programs=True)
    tokenizer = getattr(miditok, args.tokenization)(config)
    print(f"Tokenizer: {args.tokenization}, vocab size: {len(tokenizer)}")

    tokens = tokenizer(original)
    reconstructed = tokenizer(tokens)

    # Resample original to match reconstructed tpq so ticks are comparable
    orig_tpq = original.ticks_per_quarter
    recon_tpq = reconstructed.ticks_per_quarter
    if orig_tpq != recon_tpq:
        original = original.resample(tpq=recon_tpq)
        print(f"Resampled original: tpq {orig_tpq} -> {recon_tpq}")

    orig_track = original.tracks[0] if original.tracks else None
    recon_track = reconstructed.tracks[0] if reconstructed.tracks else None
    if orig_track is None:
        print("No tracks found.")
        return

    os_, oe, op, ov = notes_to_arrays(orig_track)
    rs, re, rp, rv = notes_to_arrays(recon_track)

    if args.max_ticks > 0:
        om = os_ < args.max_ticks
        rm = rs < args.max_ticks
        os_, oe, op, ov = os_[om], oe[om], op[om], ov[om]
        rs, re, rp, rv = rs[rm], re[rm], rp[rm], rv[rm]

    # -- build figure --
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.suptitle(
        f"MidiTok Tokenization Comparison \u2014 {args.tokenization}\n"
        f"{midi_path.name}  |  {len(orig_track.notes)} notes \u2192 "
        f"{len(tokens.ids)} tokens \u2192 {len(recon_track.notes)} notes  "
        f"(tpq={recon_tpq})",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.2], hspace=0.35, wspace=0.08)

    norm = plt.Normalize(0, 127)
    cmap = plt.cm.viridis
    vel_color = lambda v: cmap(norm(v))

    # Row 1: side-by-side piano rolls
    ax1 = fig.add_subplot(gs[0, 0])
    draw_piano_roll(ax1, os_, oe, op, ov, vel_color, label="Original MIDI")
    setup_pianoroll_ax(ax1, os_, oe, op)

    ax2 = fig.add_subplot(gs[0, 1])
    draw_piano_roll(ax2, rs, re, rp, rv, vel_color, label="Reconstructed (round-trip)")
    setup_pianoroll_ax(ax2, rs, re, rp)

    # Sync axis limits across pair
    xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1]))
    ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1]))
    ax1.set_xlim(xlim); ax1.set_ylim(ylim)
    ax2.set_xlim(xlim); ax2.set_ylim(ylim)

    # Row 2: overlay diff
    ax3 = fig.add_subplot(gs[1, :])
    draw_piano_roll(ax3, os_, oe, op, ov, lambda _: "#B0BEC5", alpha=0.5,
                    label="Overlay: original (gray) vs reconstructed (green/red)")

    n_cmp = min(len(os_), len(rs))
    match_mask = np.zeros(len(rs), dtype=bool)
    if n_cmp > 0:
        match_mask[:n_cmp] = (
            (os_[:n_cmp] == rs[:n_cmp]) &
            (oe[:n_cmp] == re[:n_cmp]) &
            (op[:n_cmp] == rp[:n_cmp])
        )

    if match_mask.any():
        draw_piano_roll(ax3, rs[match_mask], re[match_mask],
                        rp[match_mask], rv[match_mask],
                        lambda _: "#4CAF50", alpha=0.7)
    if (~match_mask).any():
        draw_piano_roll(ax3, rs[~match_mask], re[~match_mask],
                        rp[~match_mask], rv[~match_mask],
                        lambda _: "#F44336", alpha=0.8)

    all_s = np.concatenate([os_, rs]) if len(rs) else os_
    all_e = np.concatenate([oe, re]) if len(re) else oe
    all_p = np.concatenate([op, rp]) if len(rp) else op
    setup_pianoroll_ax(ax3, all_s, all_e, all_p)

    legend_items = [
        mpatches.Patch(color="#B0BEC5", alpha=0.5, label="Original"),
        mpatches.Patch(color="#4CAF50", alpha=0.7, label="Matched"),
        mpatches.Patch(color="#F44336", alpha=0.8, label="Difference"),
    ]
    ax3.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)

    num_matched = int(match_mask.sum())
    num_diff = len(rs) - num_matched
    pct_match = 100 * num_matched / max(len(rs), 1)
    stats = (f"Notes: {len(os_)} orig / {len(rs)} recon  |  "
             f"Matched: {num_matched} ({pct_match:.1f}%)  |  "
             f"Diff: {num_diff}")
    ax3.text(0.5, -0.08, stats, transform=ax3.transAxes,
             ha="center", fontsize=9, style="italic")

    # Row 3: token strip
    ax4 = fig.add_subplot(gs[2, :])
    tok_list = tokens.tokens if isinstance(tokens.tokens[0], str) else tokens.tokens[0]
    draw_token_strip(ax4, tok_list, max_tokens=200)

    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
