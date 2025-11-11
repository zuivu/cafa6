# Modifications for `Protein Enbeddings.ipynb`

To support both ESM and ProtTrans models, the `Protein Enbeddings.ipynb` notebook needs to be updated. Follow these steps to make it configurable.

### 1. Add a Configuration Cell

In a new cell at the top of the notebook, add the following code. This will allow you to easily switch between model types.

```python
# --- CONFIGURATION ---
MODEL_TYPE = "protrans"  # or "esm"

# --- CHOOSE YOUR MODEL ---
if MODEL_TYPE == "protrans":
    MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    MODEL_PATH = f"./embedding_model/{MODEL_NAME.replace('/', '_')}"
else: # esm
    MODEL_NAME = "esm2_t33_650M_UR50D"
    MODEL_PATH = f"./embedding_model/{MODEL_NAME}.pt"
    
FASTA_PATH = "your_sequences.fasta"
OUTPUT_DIR = "./embeddings"
```

### 2. Conditionally Load Model

Modify the model loading cell to check the `MODEL_TYPE` variable.

```python
if MODEL_TYPE == "esm":
    import esm
    # Load ESM model from the .pt file
    model_data = torch.load(MODEL_PATH)
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_data)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    print(f"ESM model '{MODEL_NAME}' loaded.")
else: # protrans
    from transformers import T5Tokenizer, T5EncoderModel
    # Load ProtTrans model from directory
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"ProtTrans model '{MODEL_NAME}' loaded.")
```

### 3. Adapt Sequence Preparation and Embedding Generation

In the processing loop, use conditional logic to handle the differences in tokenization and embedding extraction between the two model types.

```python
# Example inside your batch processing loop

if MODEL_TYPE == "esm":
    # ESM: Use the batch converter
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    
    # Extract embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_embeddings = results["representations"][33]
    
    # Generate per-sequence embeddings
    for i, (label, seq) in enumerate(zip(batch_labels, batch_strs)):
        sequence_embedding = token_embeddings[i, 1 : len(seq) + 1].mean(0)
        # ... save your embedding
        
else: # protrans
    # ProtTrans: Prepare sequences with spaces and use the tokenizer
    sequences_with_spaces = [" ".join(list(seq)) for _, seq in batch_data]
    ids = tokenizer(sequences_with_spaces, add_special_tokens=True, padding=True, return_tensors="pt")
    
    # Extract embeddings
    with torch.no_grad():
        embedding_result = model(input_ids=ids.input_ids, attention_mask=ids.attention_mask)
    
    # Detach and move to CPU
    embeddings = embedding_result.last_hidden_state.cpu()
    
    # Generate per-sequence embeddings
    for i, (label, seq) in enumerate(batch_data):
        # Correctly mask padding tokens for mean calculation
        seq_len = len(seq)
        attention_mask = ids.attention_mask[i]
        # The actual length is sum of attention mask minus special tokens
        valid_tokens = attention_mask.sum() - 2 # Exclude <bos> and <eos>
        sequence_embedding = embeddings[i, 1:valid_tokens+1].mean(dim=0)
        # ... save your embedding
```
