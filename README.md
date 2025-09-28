# Urdu Text Reuse Detection  
**Final Year Project (FYP)**  
**COMSATS University Islamabad, Lahore Campus (Session 2021–2025)**  

---

## Abstract  
Plagiarism and text reuse are rising concerns in academia, journalism, and digital content creation. While plagiarism detection systems in **high-resource languages** (e.g., English) are well established, **low-resource languages such as Urdu** remain underexplored due to limited resources and research.  

This project introduces a comprehensive framework for **Urdu Text Reuse Detection**, where we:  
1. Constructed **nine novel Urdu text reuse corpora** using a **semi-automated back-translation approach** applied to the COUNTER dataset.  
2. Applied **Machine Learning (ML)**, **Deep Learning (DL)**, and **Transformer-based models (LLMs)** to classify reuse into **Non-Derived (ND), Partially Derived (PD), and Wholly Derived (WD)**.  
3. Developed a **web-based plagiarism detection system** and **REST APIs** to make our research practically accessible.  

Our models achieved an **F1 Score of 93.4**, surpassing prior research benchmarks in Urdu plagiarism detection.  

---

## Introduction  
Text reuse involves reproducing existing text with or without modifications. In Urdu, detecting such reuse is challenging due to:  

- **Morphological richness**: Urdu words have complex inflectional patterns.  
- **Word order flexibility**: Multiple valid syntactic structures.  
- **Scarcity of resources**: No large-scale labeled datasets for plagiarism detection in Urdu.  

This project aims to **bridge the research gap** by creating reusable datasets, benchmarking models, and deploying usable systems for real-world detection.  

---

## Research Objectives  
1. **Corpus Creation**: Build **nine Urdu text reuse corpora** via **back-translation techniques** from English to Urdu and vice versa.  
2. **Classification Models**: Train ML/DL/LLM-based classifiers for three categories:  
   - Non-Derived (ND)  
   - Partially Derived (PD)  
   - Wholly Derived (WD)  
3. **System Deployment**: Implement a **web and mobile plagiarism detection system** with APIs for integration.  
4. **Contribution to Urdu NLP**: Publish corpora, models, and findings for the research community.  

---

## Problem Statement  
Despite the growing use of Urdu in academia and journalism, **reliable plagiarism detection systems are lacking**. Challenges include:  

- Absence of **large-scale Urdu text reuse datasets**.  
- Complexity of Urdu grammar, morphology, and semantics.  
- Limited prior research and weak reproducibility of existing studies.  

---

## Dataset Development  

### Source Dataset  
- **COUNTER Corpus**: A widely used dataset for text reuse and plagiarism detection in English.  

### Method  
- **Back-translation** approach:  
  1. English → Urdu (machine translation)  
  2. Urdu → English (reverse translation)  
  3. Compare and retain aligned Urdu text.  

### Output  
- **9 distinct Urdu text reuse corpora**, covering ND, PD, and WD categories.  
- Dataset sizes vary across corpora, offering **diverse training and evaluation splits**.  

---

## Methodology  

### Machine Learning (ML) Models  
- Support Vector Machines (SVM)  
- k-Nearest Neighbors (KNN)  
- Naive Bayes  

### Deep Learning (DL) Models  
- Convolutional Neural Networks (CNN)  
- Long Short-Term Memory (LSTM) networks  
- Bidirectional LSTMs (BiLSTM)  

### Transformer & LLM Approaches  
- Pretrained models from **Hugging Face** (mBERT, XLM-R, IndicBERT)  
- Fine-tuned embeddings using **Sentence Transformers**  

### System Deployment  
- **Backend**: Flask/Django-based API with ML/DL model integration.  
- **Frontend**: React.js/Next.js web app with OCR-based input support.  
- **Mobile App**: React Native implementation.  
- **Authentication**: JWT with API Key management.  
- **Hosting**: AWS / Heroku / DigitalOcean.  

---

## Experimental Setup  

- **Hardware**: NVIDIA GPU (Colab Pro / Local GPU server)  
- **Training Parameters**:  
  - Batch size: 16–32  
  - Optimizer: Adam / AdamW  
  - Learning rate: 1e-5 – 5e-5 (for Transformers)  
- **Evaluation Metrics**: Precision, Recall, F1-Score  
- **Baseline**: Compared against previous Urdu plagiarism research benchmarks  

---


## Research Contributions  
1. **Corpus Development**: Creation of **nine Urdu text reuse corpora**, the first of their kind.  
2. **Model Benchmarking**: Applied and compared ML, DL, and LLM models on Urdu text reuse tasks.  
3. **System Implementation**: Developed a **web and API-based detection system** for real-world application.  
4. **Research Impact**: Achieved state-of-the-art F1 score (93.4%) for Urdu plagiarism detection.  
5. **Community Contribution**: Resources made available for future Urdu NLP research.  

---

## Future Work  
- Expand corpora with **academic articles, news reports, and literature**.  
- Explore **cross-lingual plagiarism detection** (Urdu ↔ English).  
- Optimize LLMs for **low-resource deployment**.  
- Extend **mobile application** with offline detection.  
- Publish datasets and methods in **top-tier NLP conferences/journals**.  

---

## Related Work  
- COUNTER Corpus (baseline English reuse dataset)  
- Sentence-BERT for multilingual embeddings  
- Research in Urdu NLP (tokenization, embeddings, classification)  

---

## Contributors  

| Name | Roll No. | Contribution |
|------|----------|--------------|
| Muhammad Umer Aamir | FA21-BSE-114 | Dataset Development, ML Models |
| Malik Ashas | FA21-BSE-120 | Web System, API Development |
| Usama Tufail | FA21-BSE-053 | Research, DL/LLM Models, Documentation |

**Supervisor:**  
*Dr. Muhammad Sharjeel* (Assistant Professor, CUI Lahore)  

---

## Note  
This project **focuses exclusively on the Urdu language**.  
Cross-lingual and unsupervised approaches are considered **future extensions**.  

---

## How to Cite  
If you use our dataset, models, or system in your research, please cite:  

```bibtex
@misc{fyp2025_urdu_reuse,
  title     = {Urdu Text Reuse Detection},
  author    = {Muhammad Umer Aamir and Malik Ashas and Usama Tufail},
  year      = {2025},
  institution = {COMSATS University Islamabad, Lahore Campus},
  note      = {Final Year Project, Urdu NLP Research Contribution}
}
