# midiGPT + MidiTok REMI 训练指南

本指南说明如何在本地 GPU 环境下完成数据准备、训练和推理的完整流程。

## 环境要求

| 组件 | 要求 |
|---|---|
| GPU | NVIDIA 3080Ti 12GB（或同等级以上） |
| Python | >= 3.9 |
| PyTorch | >= 2.0（需与 CUDA 版本匹配） |
| 磁盘 | 数据集 ~5GB，模型 checkpoint ~50-200MB |

## 1. 环境安装

```bash
# 克隆 midiGPT 主仓库
git clone https://github.com/Zttt0523/midiGPT.git
cd midiGPT

# 克隆 MidiTok 仓库（获取 midigpt_upgrade 代码）
git clone https://github.com/Zttt0523/MidiTok.git
cd MidiTok
git checkout cursor/development-environment-setup-3bec
cd ..

# 安装依赖
pip install torch miditok symusic pydantic==1.* tqdm scipy ipython

# 将 upgrade 代码链接到 midiGPT
cp -r MidiTok/midigpt_upgrade midiGPT/
```

## 2. 数据集准备

### 方案 A：Maestro 数据集（推荐，古典钢琴）

```bash
# 下载 Maestro v3.0.0（约 5.3 GB）
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip -d data/maestro

# 验证
find data/maestro -name "*.midi" | wc -l
# 预期输出: 1276
```

### 方案 B：自定义 MIDI 数据集

将所有 `.mid` / `.midi` 文件放到一个目录下即可，支持嵌套子目录：

```
data/my_dataset/
├── piece1.mid
├── piece2.midi
└── subfolder/
    └── piece3.mid
```

### 方案 C：使用 MidiTok 测试数据（快速验证）

```bash
# 直接使用 MidiTok 仓库中的测试 MIDI 文件
ls MidiTok/tests/MIDIs_one_track/
```

## 3. 训练

### 3.1 推荐配置（3080Ti 12GB）

```bash
cd midiGPT/midigpt_upgrade

python train.py \
  --midi-dir ../../data/maestro \
  --output-dir output/maestro_run1 \
  --context-length 1024 \
  --embedding-size 256 \
  --num-heads 8 \
  --num-blocks 8 \
  --epochs 20 \
  --batch-size 48 \
  --lr 3e-4 \
  --eval-interval 500 \
  --generate-tokens 2048
```

**预计资源占用：**

| 指标 | 预估值 |
|---|---|
| 模型参数 | ~3.6M |
| 显存占用 | ~6-8 GB |
| 训练时长（Maestro） | ~2-4 小时 |
| 每 epoch 耗时 | ~8-12 分钟 |

### 3.2 更大模型配置（如果显存够用）

```bash
python train.py \
  --midi-dir ../../data/maestro \
  --output-dir output/maestro_large \
  --context-length 2048 \
  --embedding-size 384 \
  --num-heads 8 \
  --num-blocks 12 \
  --epochs 30 \
  --batch-size 24 \
  --lr 2e-4
```

**预计参数量：** ~11M，显存占用 ~10-11 GB。

### 3.3 快速测试配置（验证流程）

```bash
python train.py \
  --midi-dir ../../MidiTok/tests/MIDIs_one_track \
  --output-dir output/test_run \
  --context-length 256 \
  --embedding-size 64 \
  --num-heads 4 \
  --num-blocks 4 \
  --epochs 5 \
  --batch-size 64 \
  --lr 5e-4
```

### 3.4 训练参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--context-length` | 1024 | 模型能看到的最大 token 窗口，越大越能学到长程结构 |
| `--embedding-size` | 256 | 嵌入维度，必须能被 num-heads 整除 |
| `--num-heads` | 8 | 注意力头数 |
| `--num-blocks` | 8 | Transformer 层数 |
| `--batch-size` | 48 | 批大小，如果 OOM 就减小 |
| `--lr` | 3e-4 | 学习率 |
| `--epochs` | 20 | 训练轮数 |
| `--eval-interval` | 500 | 每 N 个 batch 评估一次并保存 checkpoint |
| `--max-files` | 0 | 限制使用的 MIDI 文件数量（0=全部） |
| `--valid-ratio` | 0.1 | 验证集比例 |
| `--device` | auto | 设备选择：auto / cuda / cpu / mps |

### 3.5 如果显存不足 (OOM)

按以下优先级依次降低：

1. 减小 `--batch-size`（48 → 32 → 16）
2. 减小 `--context-length`（1024 → 512）
3. 减小 `--embedding-size`（256 → 192 → 128）
4. 减小 `--num-blocks`（8 → 6）

### 3.6 训练输出

```
output/maestro_run1/
├── tokenizer.json          # 保存的 tokenizer（推理时需要）
├── checkpoints/
│   ├── best_model.ckpt     # 最佳 checkpoint
│   └── loss_history.txt    # 逐 batch 的 loss 记录
├── generated_temp0.7.mid   # 训练结束后自动生成的样本
├── generated_temp0.9.mid
└── generated_temp1.1.mid
```

## 4. 推理（生成 MIDI）

### 4.1 从零生成

```bash
python generate.py \
  --checkpoint output/maestro_run1/checkpoints/best_model.ckpt \
  --tokenizer output/maestro_run1/tokenizer.json \
  --output my_music.mid \
  --num-tokens 2048 \
  --temperature 0.9 \
  --top-k 50
```

### 4.2 从 MIDI 文件续写

给定一首 MIDI 文件，取前 N 小节作为 prompt，让模型续写：

```bash
python generate.py \
  --checkpoint output/maestro_run1/checkpoints/best_model.ckpt \
  --tokenizer output/maestro_run1/tokenizer.json \
  --prompt-midi /path/to/seed_music.mid \
  --prompt-bars 4 \
  --output continuation.mid \
  --num-tokens 2048 \
  --temperature 0.9
```

### 4.3 批量生成不同风格

```bash
for temp in 0.5 0.7 0.9 1.0 1.2 1.5; do
  python generate.py \
    --checkpoint output/maestro_run1/checkpoints/best_model.ckpt \
    --tokenizer output/maestro_run1/tokenizer.json \
    --output "samples/gen_temp${temp}.mid" \
    --num-tokens 2048 \
    --temperature $temp
done
```

### 4.4 生成参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--temperature` | 0.9 | 越低越保守/重复，越高越多样/随机 |
| `--top-k` | 50 | 只从概率最高的 k 个 token 中采样 |
| `--num-tokens` | 1024 | 生成的 token 数（~200 tokens ≈ 1-2 小节） |
| `--prompt-bars` | 4 | 续写模式下从 prompt 取多少小节 |

**temperature 经验值：**
- 0.5-0.7：保守，旋律重复但和声稳定
- 0.8-1.0：平衡，推荐起步值
- 1.1-1.5：自由，更多意外，可能出现大跳

## 5. 播放生成的 MIDI

生成的 `.mid` 文件可以用任意 MIDI 播放器打开：

- **命令行**：`timidity my_music.mid`（Linux）或 `fluidsynth my_music.mid`
- **桌面**：Windows Media Player / GarageBand / MuseScore
- **在线**：[Signal](https://signal.vercel.app/) / [midi-player](https://cifkao.github.io/html-midi-player/)
- **Python**：

```python
from midigpt_upgrade.generate import load_model_and_tokenizer
from symusic import Score

s = Score("my_music.mid")
print(f"{len(s.tracks)} tracks, {sum(len(t.notes) for t in s.tracks)} notes")
```

## 6. 常见问题

### 6.1 `pydantic` 版本冲突

midiGPT 使用 pydantic v1 API。如果遇到 `root_validator` 错误：

```bash
pip install "pydantic<2"
```

### 6.2 CUDA OOM

```
RuntimeError: CUDA out of memory
```

减小 `--batch-size` 或 `--context-length`。参见 3.5 节。

### 6.3 `torch.has_cuda` 警告

midiGPT 使用了已弃用的 `torch.has_cuda`。可以忽略此警告，不影响训练。
如果导致错误，修改 `src/midigpt/utils.py`：

```python
def get_auto_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### 6.4 生成的 MIDI 全是短音符

训练不足的表现。增加 `--epochs` 或使用更大数据集。通常在 loss < 1.5 后 duration 多样性会显著改善。
