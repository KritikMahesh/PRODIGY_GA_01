# 🤖 Text Generation with GPT-2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#scrollTo=C6KCfWcvfCsW&fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/kritikmahesh/gpt2-text-generation.fed41f83-503e-4221-9d8c-c95952ff5a91.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250813/auto/storage/goog4_request%26X-Goog-Date%3D20250813T075846Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D61e0c05f7211f3ee40be5508ca7c181aafc11dd7b6b42c94e25fb37dfc5f5cbb5b2d27653c18276646b33246038929b10d18112c9ae8fe40d6293698d76fe5af38bff3658cc4782e0c8dc15edba3cac2dbbfef80cb11d41a84dae459e0773097904ef63b840c51b3100ff1f1abf139abbc8010308c9899d728ba2de903f005afbce606496fa98ea1ef63cae2dbef67024d06cab8fca324b312b5ec7adf56a02874c84edc5ee53c1b8900573168dc716c74ba0569978e74b51613e74b003f5f977ffbd88ca338929c5392312160206b2dbf4217ed1874bf9b05de11d43c48a4bab8370753b7e4f975e9f13514270881c78a785cc03f813a103faca57d71b2c4ee)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-green.svg)](https://huggingface.co/transformers/)

> **Custom Prompt Text Generation using OpenAI’s GPT-2 Transformer Model**

This repository contains an implementation of **text generation** using **GPT-2** (124M parameters) — a powerful Transformer-based language model by OpenAI. The project demonstrates multiple decoding strategies to produce coherent, creative, and diverse text outputs from custom user prompts.

---

## 📋 Table of Contents
- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Decoding Strategies](#-decoding-strategies)
- [Training & Evaluation](#-training--evaluation)
- [Results](#-results)
- [How It Works](#-how-it-works)
- [Customization](#-customization)

---

## 🎯 About

This project showcases **prompt-based text generation** using a pre-trained **GPT-2** model. It implements **four distinct generation strategies** for different creative and control outcomes:

- **Greedy Search** – Fast, deterministic text completion  
- **Temperature Sampling** – Adds randomness for creativity  
- **Top-k Sampling** – Restricts to most probable words  
- **Top-p (Nucleus) Sampling** – Dynamically adapts vocabulary scope  

The implementation includes **quality analysis metrics** such as **perplexity** and **lexical diversity**.

---

## ✨ Features

- 🧠 **Transformer Architecture** – State-of-the-art NLP modeling  
- 🎭 **Multiple Decoding Modes** – Choose creativity or precision  
- 📊 **Quality Metrics** – Perplexity & diversity evaluation  
- 🖥️ **Interactive Interface** – Generate & compare results in real-time  
- 📦 **Hugging Face Transformers** – Easy model loading & tokenization  
- 🔧 **Customizable Parameters** – Modify temperature, max length, and more  

---

## 🛠️ Installation

**Option 1: Google Colab (Recommended)**  
1. Click the "Open in Colab" badge above  
2. Run all cells in sequence — dependencies auto-install  

**Option 2: Local Setup**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install transformers torch matplotlib numpy

# Launch Jupyter Notebook
jupyter notebook Text_Generation_with_GPT_2.ipynb
```

---

## 📊 Dataset

This project uses **pre-trained GPT-2 weights** — no large-scale training dataset is required.  
However, a **custom set of AI/technology-related prompts** is provided to test the model.

**Example Prompts:**
- `"The future of artificial intelligence is"`
- `"In a distant galaxy, humans discovered"`
- `"The key to solving climate change might be"`

---

## 🚀 Usage

### Quick Start
1. Load the notebook and select a generation strategy  
2. Enter a custom text prompt  
3. Adjust parameters (temperature, max length, top-k, top-p)  
4. Compare outputs from different decoding methods  

**Example:**
```python
prompt = "Machine learning will"
generate_text(prompt, strategy="top_p", temperature=0.9, max_length=100)
```

---

## 🏗️ Model Architecture

- **Model**: GPT-2 Small (124M parameters)  
- **Architecture**: Transformer Decoder with Multi-Head Self-Attention  
- **Layers**: 12 transformer blocks  
- **Hidden Size**: 768  
- **Heads**: 12 attention heads  
- **Positional Encoding**: Learned embeddings  

---

## 🎛️ Decoding Strategies

1. **Greedy Search** – Always picks the highest probability token  
2. **Temperature Sampling** – Scales logits to control randomness  
3. **Top-k Sampling** – Chooses from top-k most probable tokens  
4. **Top-p Sampling (Nucleus)** – Chooses from smallest probability mass ≥ p  

---

## 🎯 Training & Evaluation

- **Training**: This implementation uses a **pre-trained GPT-2**, so only fine-tuning or inference is performed  
- **Evaluation Metrics**:  
  - *Perplexity* – Measures prediction confidence  
  - *Lexical Diversity* – Evaluates vocabulary richness  

---

## 📈 Results

Sample outputs from different strategies:

| Strategy | Output Style |
|----------|--------------|
| Greedy | Deterministic, safe completions |
| Temperature=1.0 | More varied and creative |
| Top-k (k=50) | Balanced between creativity and coherence |
| Top-p (p=0.9) | Naturally flowing sentences |

---

## 🔬 How It Works

1. **Tokenization** – Convert input text into GPT-2 tokens  
2. **Generation** – Use chosen decoding strategy to predict next tokens  
3. **Detokenization** – Convert token IDs back to readable text  
4. **Analysis** – Evaluate generated text with metrics & visualizations  

---

## 🔧 Customization

### Change Model Size
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "gpt2-medium"  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
```

### Modify Generation Parameters
```python
generate_text(prompt, strategy="top_k", top_k=100, temperature=0.8, max_length=150)
```
