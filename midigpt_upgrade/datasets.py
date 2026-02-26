"""
MidiTok-powered dataset for midiGPT.

Replaces the original BachChoraleDataset with a general-purpose MIDI dataset
that uses REMI tokenization, preserving pitch + duration + position + bar structure.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from miditok import REMI, TokenizerConfig
from miditok.constants import MIDI_FILES_EXTENSIONS


def build_tokenizer(
    midi_paths: Sequence[Path],
    num_velocities: int = 16,
    use_chords: bool = True,
    use_tempos: bool = True,
    use_time_signatures: bool = True,
    use_rests: bool = False,
) -> REMI:
    """Create and return a configured REMI tokenizer."""
    config = TokenizerConfig(
        num_velocities=num_velocities,
        use_chords=use_chords,
        use_tempos=use_tempos,
        use_time_signatures=use_time_signatures,
        use_rests=use_rests,
        use_programs=True,
    )
    return REMI(config)


class MidiTokenDataset(Dataset):
    """
    General-purpose MIDI dataset using MidiTok tokenization.

    Tokenizes all MIDI files into a single concatenated token stream,
    then serves (context, target) pairs for next-token prediction.
    """

    def __init__(
        self,
        midi_paths: Sequence[Path],
        tokenizer: REMI,
        context_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.vocab_size = len(tokenizer)

        all_ids = []
        skipped = 0
        for path in midi_paths:
            try:
                tokens = tokenizer(path)
                if isinstance(tokens, list):
                    for t in tokens:
                        all_ids.extend(t.ids)
                else:
                    all_ids.extend(tokens.ids)
            except Exception:
                skipped += 1

        self.corpus = all_ids
        self.num_files = len(midi_paths) - skipped
        self.num_tokens = len(self.corpus)

    def __len__(self):
        return max(0, self.num_tokens - self.context_length)

    def __getitem__(self, idx):
        chunk = self.corpus[idx : idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def summary(self) -> str:
        return (
            f"MidiTokenDataset: {self.num_files} files, "
            f"{self.num_tokens:,} tokens, "
            f"vocab_size={self.vocab_size}, "
            f"context_length={self.context_length}"
        )


def collect_midi_files(root: Path, max_files: int = 0) -> list[Path]:
    """Recursively collect MIDI files from a directory."""
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".mid", ".midi"}
    )
    if max_files > 0:
        files = files[:max_files]
    return files
