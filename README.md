# RL_LLM_EDOS
Large Language Models with Reinforcement Learning from Human Feedback Approach for Enhancing Explainable Sexism Detection

## Overview
This repository contains the implementation of our paper on explainable sexism detection using Large Language Models (LLMs). Our work focuses on leveraging the contextual learning capabilities of LLMs to provide transparent and explainable insights into sexism detection in social media content.

## Abstract
Recent advancements in natural language processing (NLP), driven by Large Language Models (LLMs), have significantly improved text comprehension, enabling these models to handle complex tasks with greater efficiency. This research explores the application of LLMs in explainable sexism detection, utilizing a Reinforcement Learning from Human Feedback (RLHF) based fine-tuning framework. We evaluated Mistral-7B and LLaMA-3-8B models across various scenarios, achieving notable results:
- Task A (binary sexism detection): 86.811%
- Task B (category classification): 0.6829%
- Task C (fine-grained sexism vectors): 0.4722%

## Key Contributions
- Implementation of a traditional model for baseline sexism detection
- Zero-shot performance evaluation of LLMs across three EDOS tasks
- Parameter-Efficient Fine-Tuning (PEFT) mechanism implementation
- RLHF integration for enhanced category distinction and explanation quality

### Dataset
The project uses the EDOS dataset, which categorizes sexist content into three hierarchical tasks designed to train and evaluate our framework. Details on the taxonomy and dataset distribution are discussed further in the supplementary materials.

### Prerequisites
- Python 3.8 or later
- PyTorch 1.7 or later
- Transformers library by Hugging Face
  
### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/xxx/xxxx.git
cd xxxx
pip install -r requirements.txt
