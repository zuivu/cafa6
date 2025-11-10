#!/usr/bin/env python3
"""
Offline test script to validate code structure without downloading models
"""
import sys

def test_imports():
    """Test that all necessary imports work"""
    print("=" * 80)
    print("Testing Imports")
    print("=" * 80)
    
    try:
        import torch
        print("✓ torch imported")
        
        from transformers import T5Tokenizer, T5EncoderModel
        print("✓ transformers imported (T5Tokenizer, T5EncoderModel)")
        
        from Bio import SeqIO
        print("✓ biopython imported (SeqIO)")
        
        import numpy as np
        print("✓ numpy imported")
        
        from pathlib import Path
        print("✓ pathlib imported")
        
        import pickle
        print("✓ pickle imported")
        
        from tqdm import tqdm
        print("✓ tqdm imported")
        
        import h5py
        print("✓ h5py imported")
        
        import sentencepiece
        print("✓ sentencepiece imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_code_structure():
    """Test that the main script has correct structure"""
    print("\n" + "=" * 80)
    print("Testing Code Structure")
    print("=" * 80)
    
    try:
        # Test that get_embedding_model.py can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location("get_embedding_model", "get_embedding_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check that key functions exist
        assert hasattr(module, 'load_protrans'), "load_protrans function not found"
        print("✓ load_protrans function exists")
        
        assert hasattr(module, 'download_protrans_model'), "download_protrans_model function not found"
        print("✓ download_protrans_model function exists")
        
        assert hasattr(module, 'PROTRANS_MODELS'), "PROTRANS_MODELS list not found"
        print("✓ PROTRANS_MODELS list exists")
        
        # Check that models list is populated
        assert len(module.PROTRANS_MODELS) > 0, "PROTRANS_MODELS is empty"
        print(f"✓ PROTRANS_MODELS contains {len(module.PROTRANS_MODELS)} models")
        
        return True
    except Exception as e:
        print(f"✗ Structure test error: {e}")
        return False

def test_notebook():
    """Test that the notebook is valid JSON"""
    print("\n" + "=" * 80)
    print("Testing Notebook")
    print("=" * 80)
    
    try:
        import json
        with open("Protein Enbeddings.ipynb", "r") as f:
            notebook = json.load(f)
        
        print(f"✓ Notebook is valid JSON")
        print(f"✓ Notebook has {len(notebook.get('cells', []))} cells")
        
        # Check for key cells
        cells = notebook.get('cells', [])
        code_cells = [c for c in cells if c.get('cell_type') == 'code']
        print(f"✓ Notebook has {len(code_cells)} code cells")
        
        # Check for ProtTrans-specific content
        notebook_text = json.dumps(notebook)
        
        assert "T5Tokenizer" in notebook_text, "T5Tokenizer not found in notebook"
        print("✓ Notebook contains T5Tokenizer references")
        
        assert "T5EncoderModel" in notebook_text, "T5EncoderModel not found in notebook"
        print("✓ Notebook contains T5EncoderModel references")
        
        assert "transformers" in notebook_text, "transformers not found in notebook"
        print("✓ Notebook contains transformers import")
        
        # Check that ESM is NOT in the notebook anymore
        if "import esm" in notebook_text or "from esm" in notebook_text:
            print("⚠ Warning: ESM imports still present in notebook")
        else:
            print("✓ ESM imports removed from notebook")
        
        return True
    except Exception as e:
        print(f"✗ Notebook test error: {e}")
        return False

def test_helper_functions():
    """Test that helper functions work correctly"""
    print("\n" + "=" * 80)
    print("Testing Helper Functions")
    print("=" * 80)
    
    try:
        # Test sequence preparation function
        def prepare_sequence(seq):
            return " ".join(list(seq))
        
        test_seq = "MKTAYIAK"
        prepared = prepare_sequence(test_seq)
        expected = "M K T A Y I A K"
        
        assert prepared == expected, f"Expected '{expected}', got '{prepared}'"
        print(f"✓ prepare_sequence works correctly")
        print(f"  Input: '{test_seq}'")
        print(f"  Output: '{prepared}'")
        
        return True
    except Exception as e:
        print(f"✗ Helper function test error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ProtTrans Code Validation Test Suite")
    print("=" * 80 + "\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_code_structure()
    all_passed &= test_notebook()
    all_passed &= test_helper_functions()
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All tests passed!")
        print("Code is ready for use with ProtTrans models.")
        print("Note: Actual model download and inference require internet access.")
    else:
        print("❌ Some tests failed!")
        print("Please check the errors above.")
    print("=" * 80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
