# Migration from ESM to ProtTrans

This document describes the changes made to adapt the codebase from ESM to ProtTrans models.

## Overview

The codebase has been migrated from using Facebook's ESM (Evolutionary Scale Modeling) to ProtTrans models from Rostlab/Hugging Face. ProtTrans models are based on T5 architecture and provide state-of-the-art protein embeddings.

## Key Changes

### 1. Dependencies (`requirements.txt`)
- **Removed**: `fair-esm` package
- **Added**: 
  - `transformers>=4.30.0` - Hugging Face transformers library
  - `sentencepiece>=0.1.99` - Required for T5 tokenization
  - `protobuf>=3.20.0` - Required for model serialization

### 2. Model Download Script (`get_embedding_model.py`)

#### Before (ESM):
```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
```

#### After (ProtTrans):
```python
from transformers import T5Tokenizer, T5EncoderModel
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
```

### 3. Notebook Changes (`Protein Enbeddings.ipynb`)

#### Sequence Preprocessing
ProtTrans models require spaces between amino acids:
```python
def prepare_sequence(seq):
    return " ".join(list(seq))  # "MKTAY" -> "M K T A Y"
```

#### Batch Conversion
- **Before**: Used ESM's `alphabet.get_batch_converter()`
- **After**: Use T5 tokenizer directly with `tokenizer(..., return_tensors="pt")`

#### Embedding Extraction
- **Before**: ESM returns embeddings with specific structure
- **After**: T5 returns `last_hidden_state` that needs mean pooling

### 4. Available Models

#### ESM Models (Old):
- esm2_t6_8M_UR50D
- esm2_t12_35M_UR50D
- esm2_t33_650M_UR50D
- esm2_t36_3B_UR50D

#### ProtTrans Models (New):
- Rostlab/prot_t5_xl_uniref50 (default, recommended)
- Rostlab/prot_t5_xl_bfd
- Rostlab/prot_t5_xxl_uniref50 (larger)
- Rostlab/prot_t5_xxl_bfd
- Rostlab/prot_bert
- Rostlab/prot_bert_bfd

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Download Model
```bash
python get_embedding_model.py --model_name Rostlab/prot_t5_xl_uniref50 --model_dir ./embedding_model
```

### Using the Notebook
1. Open `Protein Enbeddings.ipynb`
2. Update the `fasta_path` and `output_dir` variables
3. Run all cells to extract embeddings

## Batch Sizes

ProtTrans models are typically larger than ESM models, so batch sizes have been adjusted:
- Short sequences (≤100 aa): 32 (was 64)
- Medium sequences (≤500 aa): 8 (was 16)
- Long sequences (≤1000 aa): 4 (was 8)
- Very long sequences: 2 or 1

## Performance Notes

- ProtTrans models may be slower than ESM due to larger model size
- First run will download the model (~1-3 GB depending on variant)
- GPU is highly recommended for processing large datasets
- Mean pooling is used to get per-sequence embeddings

## Compatibility

The output format (pickle and npz files) remains the same, so downstream analysis code should work without modification.

## Testing

Run the offline validation test:
```bash
python test_offline.py
```

This verifies:
- All dependencies are installed
- Code structure is correct
- Notebook is valid
- Helper functions work

## References

- ProtTrans paper: https://ieeexplore.ieee.org/document/9477085
- Hugging Face models: https://huggingface.co/Rostlab
- Transformers library: https://huggingface.co/docs/transformers
