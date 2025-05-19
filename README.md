# Math Problem Type Classification with Qwen3 & Unsloth 🧠➗

This project fine-tunes the [Qwen3 model](https://huggingface.co/Qwen/Qwen1.5-4B) using the [Unsloth](https://github.com/unslothai/unsloth) framework to classify **math word problems** into categories. The dataset comes from the [Kaggle: Classification of Math Problems](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy) competition.

We adapt the base model for **text classification** by trimming the language modeling head and applying Hugging Face's `SFTTrainer`.

---

## 🧠 Task Overview

> Given a math word problem, predict its **problem type** from a set of labeled categories.

- Multi-class classification task
- Dataset: Word problems labeled by topic (e.g., algebra, geometry, arithmetic)
- Goal: Train a Qwen3-based model to predict the correct category

---

## 🔧 Key Features

- ✅ Fine-tuning Qwen3 using `unsloth.FastLanguageModel`
- ✅ LoRA adapters for memory-efficient training
- ✅ Kaggle dataset integration
- ✅ Hugging Face `trl` library for training (`SFTTrainer`)
- ✅ Data preprocessing and output logits trimming for classification

---

## 📂 Notebook Workflow

1. **Setup**
   - Installs required packages
   - Loads Kaggle dataset and sets up API keys

2. **Model Loading**
   - Loads `Qwen3-4B` with Unsloth
   - Adds classification-specific head by trimming vocab to `[0, NUM_CLASSES]`

3. **Data Preparation**
   - Preprocesses dataset into text–label pairs
   - Tokenizes and converts to Hugging Face `Dataset` format

4. **Fine-Tuning**
   - Uses Hugging Face `SFTTrainer`
   - Fine-tunes with LoRA adapters on classification labels

5. **Validation and Testing**
   - Evaluates on separate test and validation sets

---

## 🚀 Quick Start

1. Place your `kaggle.json` API key in the working directory
2. Open `Mathprob_qwen_unsloth_classification.ipynb`
3. Run all cells in order

---

## 🛠️ Requirements

- Python 3.10+
- Torch 2.1+ with GPU
- kaggle CLI & API key
- Hugging Face libraries (`trl`, `transformers`, etc.)

---

## 🧩 Dataset Info

- [Kaggle Competition Page](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy)
- Dataset contains math problems and their categories

---

## 📦 Libraries Used

- `unsloth`
- `transformers`
- `datasets`
- `trl`
- `kaggle`

---

## 🙏 Acknowledgements

- [Qwen Model](https://huggingface.co/Qwen)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Kasut Academy](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy)
