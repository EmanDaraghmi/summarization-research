# Transformer-based Summarization Research Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-🤗-yellow.svg)](https://huggingface.co)

A comprehensive research framework for evaluating transformer-based summarization models across multiple datasets with rigorous experimental methodology and full reproducibility.

## 📋 Overview

This repository implements a systematic research methodology for evaluating state-of-the-art transformer-based summarization models (PEGASUS, BART, T5) across three benchmark datasets (CNN/DailyMail, XSum, SamSum). The framework addresses four critical dimensions: dataset preparation, model configuration, experimental protocol, and evaluation strategy.

## 🎯 Key Features

- **Two-Phase Experimental Design**: Phase 1 (~6% validation+test) and Phase 2 (80/10/10 split)
- **Multiple Models**: PEGASUS, BART, T5-Small, and T5-Base evaluation
- **Diverse Datasets**: CNN/DailyMail (news), XSum (abstractive), SamSum (conversational)
- **Statistical Rigor**: Paired bootstrap resampling with 1000 samples (α=0.05)
- **Full Reproducibility**: Fixed random seeds, containerized environment, complete documentation
- **Comprehensive Evaluation**: ROUGE metrics, training efficiency, and human evaluation protocols

## 📊 Research Methodology

### Experimental Design
- **Phase 1**: Original dataset splits (~6% validation+test)
- **Phase 2**: Updated 80/10/10 stratified splits for robust evaluation
- **Stratified Sampling**: Maintains original distribution of summary lengths and content types

### Models Evaluated
| Model | Parameters | Max Length | Specialization |
|-------|------------|------------|----------------|
| PEGASUS-Large | 568M | 1024 | News summarization (GSG pre-training) |
| BART-Large | 406M | 1024 | Abstractive summarization |
| T5-Small | 60M | 512 | Lightweight baseline |
| T5-Base | 220M | 512 | Balanced efficiency-accuracy |

### Datasets
| Dataset | Content Type | Avg Source Length | Summary Style | Size |
|---------|-------------|-------------------|---------------|------|
| CNN/DailyMail | News articles | 781 tokens | Multi-sentence highlights | 286K/13K/11K |
| XSum | BBC articles | 431 tokens | Single-sentence abstractive | 204K/11K/11K |
| SamSum | Dialogues | 124 tokens | Single-paragraph | 14K/818/819 |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip
- Git

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/EmanDaraghmi/summarization-research.git
cd summarization-research

# Run automated setup script
./setup.sh

# Activate virtual environment
source research-env/bin/activate
