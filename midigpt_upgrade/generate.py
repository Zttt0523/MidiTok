"""
MIDI generation pipeline for midiGPT.

Loads a trained model checkpoint, generates token sequences,
and converts them back to playable MIDI files.

Usage:
    python generate.py --checkpoint output/checkpoints/best_model.ckpt \
                       --tokenizer output/tokenizer.json \
                       --num-tokens 2048 --temperature 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from midigpt import GPT
from midigpt.config import ModelConfigure
from miditok import REMI, TokSequence


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: str = "auto",
) -> tuple[GPT, REMI, str]:
    """Load a trained model and its associated tokenizer."""
    if device == "auto":
        device = get_device()

    tokenizer = REMI(params=tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device
    model = GPT(ModelConfigure(**config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model: {model.num_params:,} params, loss={checkpoint['loss']:.4f}")
    print(f"Device: {device}")

    return model, tokenizer, device


def generate_midi(
    model: GPT,
    tokenizer: REMI,
    output_path: Path,
    seed_ids: list[int] | None = None,
    num_tokens: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
) -> Path:
    """Generate a MIDI file from the model."""
    if seed_ids is None:
        bar_token = "Bar_None"
        seed_ids = [tokenizer.vocab.get(bar_token, 0)]

    generated_ids = model.generate(
        idx=seed_ids,
        num_generated_tokens=num_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
        as_list=True,
    )

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
    num_tokens: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
) -> Path:
    """Generate a MIDI continuation from an existing MIDI file prompt."""
    tokens = tokenizer(prompt_midi_path)
    if isinstance(tokens, list):
        tokens = tokens[0]

    bar_count = 0
    cutoff = len(tokens.ids)
    for i, tok in enumerate(tokens.tokens):
        if tok == "Bar_None":
            bar_count += 1
            if bar_count > prompt_bars:
                cutoff = i
                break

    seed_ids = tokens.ids[:cutoff]
    print(f"Prompt: {len(seed_ids)} tokens ({bar_count - 1} bars) from {prompt_midi_path.name}")

    generated_ids = model.generate(
        idx=seed_ids,
        num_generated_tokens=num_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
        as_list=True,
    )

    tok_seq = TokSequence(ids=generated_ids)
    midi = tokenizer.decode(tok_seq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump_midi(str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI with trained midiGPT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated.mid")
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt-midi", type=str, default=None,
                        help="MIDI file to use as prompt (continuation mode)")
    parser.add_argument("--prompt-bars", type=int, default=4,
                        help="Number of bars to take from prompt MIDI")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        Path(args.checkpoint), Path(args.tokenizer), args.device,
    )

    output_path = Path(args.output)

    if args.prompt_midi:
        path = generate_from_midi_prompt(
            model, tokenizer,
            Path(args.prompt_midi), output_path,
            prompt_bars=args.prompt_bars,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    else:
        path = generate_midi(
            model, tokenizer, output_path,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    from symusic import Score
    s = Score(str(path))
    notes = sum(len(t.notes) for t in s.tracks)
    print(f"Generated: {notes} notes, {len(s.tracks)} tracks -> {path}")


if __name__ == "__main__":
    main()
