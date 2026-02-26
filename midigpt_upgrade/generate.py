"""
MIDI generation pipeline for midiGPT.

Loads a trained model checkpoint, generates token sequences,
and converts them back to playable MIDI files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path("/tmp/midiGPT/src")))

from midigpt import GPT, ModelConfigure
from miditok import REMI


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
) -> tuple[GPT, REMI]:
    """Load a trained model and its associated tokenizer."""
    tokenizer = REMI(params=tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = GPT(ModelConfigure(**config))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_midi(
    model: GPT,
    tokenizer: REMI,
    output_path: Path,
    seed_ids: Optional[list[int]] = None,
    num_tokens: int = 512,
    temperature: float = 0.9,
    top_k: int = 40,
) -> Path:
    """
    Generate a MIDI file from the model.

    Args:
        seed_ids: initial token ids to seed the generation. If None, starts
                  with a BOS-like token (first Bar token).
        num_tokens: number of tokens to generate.
        temperature: sampling temperature (higher = more creative).
        top_k: top-k sampling parameter.
    """
    if seed_ids is None:
        bar_token = "Bar_None"
        if bar_token in tokenizer.vocab:
            seed_ids = [tokenizer.vocab[bar_token]]
        else:
            seed_ids = [0]

    generated_ids = model.generate(
        idx=seed_ids,
        num_generated_tokens=num_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
        as_list=True,
    )

    from miditok import TokSequence
    tok_seq = TokSequence(ids=generated_ids)
    midi = tokenizer.decode(tok_seq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump_midi(str(output_path))
    return output_path


def generate_from_midi_prompt(
    model: GPT,
    tokenizer: REMI,
    prompt_midi_path: Path,
    output_path: Path,
    prompt_bars: int = 4,
    num_tokens: int = 512,
    temperature: float = 0.9,
    top_k: int = 40,
) -> Path:
    """
    Generate a MIDI continuation from an existing MIDI file prompt.

    Takes the first `prompt_bars` bars of the prompt MIDI as seed,
    then generates `num_tokens` additional tokens.
    """
    tokens = tokenizer(prompt_midi_path)
    if isinstance(tokens, list):
        tokens = tokens[0]

    bar_token = "Bar_None"
    bar_count = 0
    cutoff = len(tokens.ids)
    for i, tok in enumerate(tokens.tokens):
        if tok == bar_token:
            bar_count += 1
            if bar_count > prompt_bars:
                cutoff = i
                break

    seed_ids = tokens.ids[:cutoff]

    generated_ids = model.generate(
        idx=seed_ids,
        num_generated_tokens=num_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
        as_list=True,
    )

    from miditok import TokSequence
    tok_seq = TokSequence(ids=generated_ids)
    midi = tokenizer.decode(tok_seq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump_midi(str(output_path))
    return output_path
