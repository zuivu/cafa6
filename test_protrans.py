#!/usr/bin/env python3
"""
Test script to validate ProtTrans model functionality
"""
import torch
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np

def prepare_sequence(seq):
    """Add space between each amino acid for ProtTrans"""
    return " ".join(list(seq))

def test_protrans_embedding():
    """Test basic embedding extraction with a small model"""
    print("=" * 80)
    print("Testing ProtTrans Model Functionality")
    print("=" * 80)
    
    # Test with a smaller, faster model for validation
    model_name = "Rostlab/prot_t5_xl_uniref50"
    
    print(f"\n1. Loading model: {model_name}")
    print("   (This may take a few minutes on first run...)")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        print("   ✓ Tokenizer loaded")
        
        model = T5EncoderModel.from_pretrained(model_name)
        print("   ✓ Model loaded")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"   ✓ Model moved to {device}")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False
    
    # Test sequences
    test_sequences = [
        ("test_seq_1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"),
        ("test_seq_2", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
    ]
    
    print(f"\n2. Testing with {len(test_sequences)} sequences")
    
    for seq_id, seq in test_sequences:
        print(f"\n   Processing: {seq_id}")
        print(f"   Length: {len(seq)} amino acids")
        
        try:
            # Prepare sequence
            prepared_seq = prepare_sequence(seq)
            
            # Tokenize
            ids = tokenizer([prepared_seq], add_special_tokens=True, padding="longest", return_tensors="pt")
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)
            
            # Extract embedding
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
            
            # Mean pooling
            seq_len = attention_mask[0].sum().item()
            seq_embedding = embeddings[0, :seq_len].mean(dim=0).cpu().numpy()
            
            print(f"   ✓ Embedding shape: {seq_embedding.shape}")
            print(f"   ✓ Embedding stats: mean={seq_embedding.mean():.4f}, std={seq_embedding.std():.4f}")
            
        except Exception as e:
            print(f"   ✗ Error processing sequence: {e}")
            return False
    
    print("\n" + "=" * 80)
    print("✅ All tests passed successfully!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_protrans_embedding()
    exit(0 if success else 1)
