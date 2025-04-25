# LLM4MS: Large Language Model Derived Spectral Embeddings

## 1. Introduction

LLMs for improving compound identification in electron ionization mass spectrometry (EI-MS).

## 2. Data

### Reference Library (`lib`)

The reference library embeddings provided or used by scripts in this repository correspond to the **million-scale in-silico EI-MS library** described in:

> Yang, Q., Ji, H., Xu, Z. et al. Ultra-fast and accurate electron ionization mass spectrum matching for compound identification with million-scale in-silico library. *Nat Commun* **14**, 3722 (2023). https://doi.org/10.1038/s41467-023-39279-7

The actual spectral data for this library is publicly available from the authors of the cited paper via Zenodo (accession code 7476120). Our repository may contain the pre-computed LLM4MS *embeddings* for this library.

### Query Data (`query_small` / `query_full`)

* **`query_small`**: This directory contains a small subset of query spectra (e.g., in a processed format suitable for direct use with our scripts) derived from the **NIST 2023 Tandem Mass Spectral Library (mainlib)**. This subset is provided for quick testing and demonstration purposes. Please note that access to the complete NIST 2023 library requires obtaining the appropriate license from NIST.
* **`query_full`**: This file contains the **SMILES strings** for all 9,921 query compounds used for benchmarking in our paper. These compounds correspond to spectra present in the NIST 2023 library (mainlib). Users need appropriate access to the NIST 2023 library to obtain the actual spectral data corresponding to these SMILES.

## 3. Models

### Base LLM (Llama)

The LLM4MS model provided in this repository is based on **Llama 3.1-8B**.
* **Permissions:** Access to the original Llama 3.1-8B model weights requires adherence to Meta's license agreement. Users typically need to request access via resources like Hugging Face.
* **Fine-tuning:** The LLM4MS model results from a specific fine-tuning procedure applied to the base Llama model, as detailed in **LLM2Vec** paper.

### LLM2Vec Preliminary Training

The preliminary training stages (MNTP, SimCSE) used to enhance the base LLM's embedding capabilities follow the methodology described in the **LLM2Vec** paper:

> BehnamGhader, P., Adlakha, V., Mosbach, M. et al. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. *arXiv preprint arXiv:2404.05961* (2024).

Users who have obtained necessary permissions for the base Llama model can utilize the resulting LLM4MS model (either provided here or reproducible via scripts) for spectral matching tasks and further research.

## 4. Software Tool

We provide a user-friendly standalone software tool with a Graphical User Interface (GUI) for easy application of LLM4MS. This tool allows users to load query spectra and perform rapid matching against the pre-computed embeddings of the in-silico library, even without extensive computational expertise.

The software can be downloaded from:
**[Link to your software download - replace xxx]**

