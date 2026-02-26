"""
Training script: midiGPT + MidiTok REMI tokenization.

Trains on MIDI files from any directory with proper GPU support.

Usage:
    python train.py --midi-dir /path/to/midis [options]

See GUIDE.md for full documentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from midigpt import TrainConfigure, Trainer
from midigpt.gpt import GPT

from datasets import MidiTokenDataset, build_tokenizer, collect_midi_files
from generate import generate_midi


def get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU detected: {name} ({mem:.1f} GB)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) detected")
        return "mps"
    print("No GPU detected, using CPU (training will be slow)")
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train midiGPT with MidiTok REMI")
    parser.add_argument("--midi-dir", type=str, required=True,
                        help="Directory containing MIDI files for training")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit number of MIDI files (0 = all)")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--generate-tokens", type=int, default=1024,
                        help="Number of tokens to generate after training")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                        help="Fraction of files for validation (default: 0.1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device() if args.device == "auto" else args.device

    # ---- Step 1: Collect MIDI files ----
    midi_dir = Path(args.midi_dir)
    files = collect_midi_files(midi_dir, max_files=args.max_files)
    print(f"\nFound {len(files)} MIDI files in {midi_dir}")

    if len(files) == 0:
        print("No MIDI files found. Exiting.")
        return

    split_idx = max(1, int(len(files) * (1 - args.valid_ratio)))
    train_files = files[:split_idx]
    valid_files = files[split_idx:]
    print(f"Split: {len(train_files)} train, {len(valid_files)} validation")

    # ---- Step 2: Build tokenizer ----
    tokenizer = build_tokenizer()
    vocab_size = len(tokenizer)
    print(f"Tokenizer: REMI, vocab_size={vocab_size}")

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # ---- Step 3: Build datasets ----
    print("\nTokenizing MIDI files (this may take a few minutes)...")
    train_dataset = MidiTokenDataset(train_files, tokenizer, context_length=args.context_length)
    print(f"Train: {train_dataset.summary()}")

    valid_dataset = None
    if valid_files:
        valid_dataset = MidiTokenDataset(valid_files, tokenizer, context_length=args.context_length)
        print(f"Valid: {valid_dataset.summary()}")

    if len(train_dataset) == 0:
        print("Training dataset is empty after tokenization. Exiting.")
        return

    # ---- Step 4: Configure and train ----
    config = TrainConfigure(
        vocab_size=vocab_size,
        context_length=args.context_length,
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        attn_dropout_prob=0.1,
        embed_dropout_prob=0.1,
        eval_interval=args.eval_interval,
        checkpoint_path=str(output_dir / "checkpoints"),
        device=device,
    )

    trainer = Trainer(config)
    num_params = trainer.model.num_params
    print(f"\nModel: {num_params:,} parameters")
    print(f"Config: embed={args.embedding_size}, heads={args.num_heads}, "
          f"blocks={args.num_blocks}, ctx={args.context_length}")
    print(f"Device: {device}")
    print(f"Training for {args.epochs} epochs...\n")

    trainer.train(train_dataset, validation_dataset=valid_dataset)

    # ---- Step 5: Generate samples ----
    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)

    checkpoint_path = output_dir / "checkpoints" / "best_model.ckpt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]
        from midigpt.config import ModelConfigure
        model = GPT(ModelConfigure(**{**model_config, "device": device}))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        for temp in [0.7, 0.9, 1.1]:
            midi_path = output_dir / f"generated_temp{temp}.mid"
            generate_midi(
                model, tokenizer, midi_path,
                num_tokens=args.generate_tokens,
                temperature=temp,
                top_k=50,
            )
            from symusic import Score
            s = Score(str(midi_path))
            notes = sum(len(t.notes) for t in s.tracks)
            print(f"  temp={temp}: {notes} notes -> {midi_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
