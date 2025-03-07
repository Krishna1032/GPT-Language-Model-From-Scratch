# GPT-Language-Model

This repository contains an implementation of a character-level Transformer-based language model, inspired by the "Attention Is All You Need" paper (2017) and Andrej Karpathy's "Build LLM from Scratch" series. The model is a decoder-only architecture, following the original Transformer design while omitting the encoder. Future updates will incorporate insights from Sebastian Raschka's "Build LLM from Scratch" book.

---

<p align="center">
    <img src="img/transformer_diagram.png" alt="Transformer Model Architecture" width="500">
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Future Improvements](#future-improvements)
- [References](#references)

## Introduction

This project implements a GPT-style language model from scratch using PyTorch. The model is trained on the Tiny Shakespeare dataset and operates at the character level. It generates text by predicting the next character in a sequence, leveraging self-attention and positional encoding to capture context.

## Features

- Character-level language modeling
- Decoder-only Transformer architecture
- Multi-head self-attention mechanism
- Positional embeddings
- Feedforward layers with ReLU activation
- Layer normalization
- Dropout for regularization
- Training using AdamW optimizer
- Ability to generate text given an initial context

## Installation

To use this implementation, install the required dependencies:

```bash
pip install torch requests
```

## Usage

Clone the repository and run the training script:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
python train.py
```

After training, it automatically generates text with:

```python
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_tokens=500)[0].tolist()))
```

## Model Architecture

This implementation follows the decoder-only architecture of the Transformer model:

1. **Token and Positional Embeddings**: Converts input characters into dense vector representations.
2. **Self-Attention Mechanism**: Computes attention weights to capture dependencies between tokens.
3. **Multi-Head Attention**: Splits the embeddings into multiple heads to learn different relationships.
4. **Feedforward Network**: Processes the attended embeddings through two linear layers with ReLU activation.
5. **Layer Normalization & Dropout**: Stabilizes training and prevents overfitting.

### Transformer Block Diagram

Below is the transformer architecture used in this implementation:

## Training

- The model is trained on the Tiny Shakespeare dataset (~1MB of text).
- Uses a batch size of 32 and sequence length (block size) of 128.
- Optimized using AdamW with a learning rate of 3e-3.
- Training runs for 5000 iterations with periodic evaluation.

## Future Improvements

- Implement Byte-Pair Encoding (BPE) for subword tokenization.
- Introduce rotary embeddings instead of learned positional embeddings.
- Experiment with larger datasets for improved text generation.
- Add temperature scaling for more diverse text generation.
- Implement FlashAttention for faster training.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Andrej Karpathy's "Build LLM from Scratch" YouTube series
- Sebastian Raschka's "Build LLM from Scratch"
- PyTorch documentation

This project serves as a foundation for understanding transformer-based language models. Contributions and suggestions are welcome!
