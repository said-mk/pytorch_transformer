# Transformer (Attention Is All You Need)

This repository contains a readable implementation of the original Transformer model from scratch in PyTorch, as described in the paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/pdf/1706.03762).

## Overview

- Implements the encoder-decoder Transformer architecture from scratch using PyTorch.
- Follows the structure and notation of the original paper for clarity and educational value.
- Dataset loading directly from Hugging Face Hub (Parquet formats).
- Tokenization using Hugging Face `tokenizers`.
- Training script with TensorBoard logging, model checkpointing, and validation loops (Character Error Rate & BLEU score).

## Structure

- `model.py`: Core Transformer model and components (Multi-head attention, Feed-Forward, Positional Encoding, etc.)
- `dataset.py`: PyTorch Dataset implementation and ByteLevelBPETokenizer generation.
- `train.py`: Training script with metric tracking.
- `config.py`: Training hyperparameters and path configurations.
- `colab_train.ipynb`: Transformer model training notebook designed specifically for Google Colab. Saves model checkpoints and vocab files in case of runtime disconnects.

## Usage

### Local Training
1. Clone the repository:
   ```bash
   git clone https://github.com/said-mk/pytorch_transformer.git
   cd pytorch_transformer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python train.py
   ```

### Google Colab Training
To train this model on Google Colab without losing progress due to random disconnects:
1. Upload or open the `colab_train.ipynb` file in Google Colab.
2. The notebook will automatically mount your Google Drive, set up the necessary absolute paths, and write checkpoints, TensorBoard logs, and vocab files directly to your Drive!
3. If disconnected, simply rerun the notebook and it will automatically preload the latest weights and resume training exactly where it left off.

## References
- [Original Paper (arXiv)](https://arxiv.org/pdf/1706.03762)
- [Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Umar Jamil (YouTube)](https://www.youtube.com/watch?v=ISNdQcPhsts)

---

Feel free to use, modify, or extend this code for research and educational purposes.
