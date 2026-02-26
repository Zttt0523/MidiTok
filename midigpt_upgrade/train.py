"""
Training script: midiGPT + MidiTok REMI tokenization.

Trains on MIDI files from any directory. Demonstrates with MidiTok test corpus
(classical piano), but can be pointed at any MIDI dataset.

Usage:
    python train.py [--midi-dir path/to/midis] [--epochs 10] [--context-length 512]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/tmp/midiGPT/src")))

from midigpt import TrainConfigure, Trainer
from midigpt.gpt import GPT

from datasets import MidiTokenDataset, build_tokenizer, collect_midi_files
from generate import generate_midi


def main():
    parser = argparse.ArgumentParser(description="Train midiGPT with MidiTok REMI")
    parser.add_argument("--midi-dir", type=str, default="/workspace/tests/MIDIs_one_track")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="/workspace/midigpt_upgrade/output")
    parser.add_argument("--generate-tokens", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Collect MIDI files ----
    midi_dir = Path(args.midi_dir)
    files = collect_midi_files(midi_dir, max_files=args.max_files)
    print(f"Found {len(files)} MIDI files in {midi_dir}")

    if len(files) == 0:
        print("No MIDI files found. Exiting.")
        return

    # Split: 80% train, 20% validation
    split_idx = max(1, int(len(files) * 0.8))
    train_files = files[:split_idx]
    valid_files = files[split_idx:]
    print(f"Split: {len(train_files)} train, {len(valid_files)} validation")

    # ---- Step 2: Build tokenizer ----
    tokenizer = build_tokenizer(train_files)
    vocab_size = len(tokenizer)
    print(f"Tokenizer: REMI, vocab_size={vocab_size}")

    # Save tokenizer for later use in generation
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # ---- Step 3: Build datasets ----
    train_dataset = MidiTokenDataset(train_files, tokenizer, context_length=args.context_length)
    print(f"Train: {train_dataset.summary()}")

    valid_dataset = None
    if valid_files:
        valid_dataset = MidiTokenDataset(valid_files, tokenizer, context_length=args.context_length)
        print(f"Valid: {valid_dataset.summary()}")

    if len(train_dataset) == 0:
        print("Training dataset is empty. Exiting.")
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
        eval_interval=200,
        checkpoint_path=str(output_dir / "checkpoints"),
        device="cpu",
    )

    trainer = Trainer(config)
    num_params = trainer.model.num_params
    print(f"\nModel: {num_params:,} parameters")
    print(f"Config: embed={args.embedding_size}, heads={args.num_heads}, "
          f"blocks={args.num_blocks}, ctx={args.context_length}")
    print(f"Training for {args.epochs} epochs...\n")

    trainer.train(train_dataset, validation_dataset=valid_dataset)

    # ---- Step 5: Generate a sample ----
    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)

    checkpoint_path = output_dir / "checkpoints" / "best_model.ckpt"
    if checkpoint_path.exists():
        checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]
        from midigpt.config import ModelConfigure
        model = GPT(ModelConfigure(**model_config))
        model.load_state_dict(checkpoint["model_state_dict"])
        device = "cpu"
        model.to(device)
        model.eval()

        for temp in [0.8, 1.0, 1.2]:
            midi_path = output_dir / f"generated_temp{temp}.mid"
            generate_midi(
                model, tokenizer, midi_path,
                num_tokens=args.generate_tokens,
                temperature=temp,
                top_k=40,
            )
            print(f"Generated: {midi_path} (temperature={temp})")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
