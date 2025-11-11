import argparse
import os
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5EncoderModel
import esm

# Valid Model Names
PROTRANS_MODELS = [
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/prot_t5_xl_bfd",
    "Rostlab/prot_t5_xxl_uniref50",
    "Rostlab/prot_t5_xxl_bfd",
    "Rostlab/prot_bert",
    "Rostlab/prot_bert_bfd",
]

ESM2_MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]

def load_protrans(model_name: str):
    """Load ProtTrans model and tokenizer from Hugging Face."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    print(f"Loading model {model_name}...")
    model = T5EncoderModel.from_pretrained(model_name)
    
    return model, tokenizer

def load_esm2(model_name: str):
    """Loads ESM-2 model and alphabet."""
    if hasattr(esm, "pretrained") and hasattr(esm.pretrained, model_name):
        return getattr(esm.pretrained, model_name)()
    if hasattr(esm.pretrained, "load_model_and_alphabet_hub"):
        return esm.pretrained.load_model_and_alphabet_hub(model_name)
    if hasattr(esm.pretrained, "load_model_and_alphabet"):
        return esm.pretrained.load_model_and_alphabet(model_name)
    raise RuntimeError(
        f"Cannot load '{model_name}'. Your installed 'fair-esm' lacks known loaders."
    )

def download_model(model_type: str, model_name: str, save_dir: Path) -> None:
    """Download a model to a specific directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name} ({model_type}) to {save_dir}...")
    print("This may take a while depending on model size and internet speed...")

    if model_type == 'protrans':
        os.environ['TRANSFORMERS_CACHE'] = str(save_dir)
        os.environ['HF_HOME'] = str(save_dir)
        model, tokenizer = load_protrans(model_name)
        
        model_save_path = save_dir / model_name.replace("/", "_")
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"Model downloaded and saved to: {model_save_path}")
        total_size = sum(f.stat().st_size for f in model_save_path.rglob('*') if f.is_file())
        print(f"Total size: {total_size / (1024**3):.2f} GB")

    elif model_type == 'esm':
        os.environ['TORCH_HOME'] = str(save_dir)
        model, alphabet = load_esm2(model_name)
        
        model_path = save_dir / f"{model_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
        }, model_path)
        
        print(f"Model downloaded and saved to: {model_path}")
        print(f"File size: {model_path.stat().st_size / (1024**3):.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Download a protein embedding model.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['esm', 'protrans'],
        help="Type of the model to download ('esm' or 'protrans')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model variant",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("./embedding_model"),
        help="Directory where to save the model",
    )
    args = parser.parse_args()

    if args.model_type == 'esm' and args.model_name not in ESM2_MODELS:
        raise ValueError(f"Invalid ESM model name. Choose from: {ESM2_MODELS}")
    if args.model_type == 'protrans' and args.model_name not in PROTRANS_MODELS:
        raise ValueError(f"Invalid ProtTrans model name. Choose from: {PROTRANS_MODELS}")

    download_model(args.model_type, args.model_name, args.model_dir)

if __name__ == "__main__":
    main()
