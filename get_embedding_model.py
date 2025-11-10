import argparse
import os
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5EncoderModel


# Valid ProtTrans model names
PROTRANS_MODELS = [
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/prot_t5_xl_bfd",
    "Rostlab/prot_t5_xxl_uniref50",
    "Rostlab/prot_t5_xxl_bfd",
    "Rostlab/prot_bert",
    "Rostlab/prot_bert_bfd",
]

def load_protrans(model_name: str):
    """
    Load ProtTrans model and tokenizer from Hugging Face.
    
    Args:
        model_name: Name of the ProtTrans model on Hugging Face
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    print(f"Loading model {model_name}...")
    model = T5EncoderModel.from_pretrained(model_name)
    
    return model, tokenizer


def download_protrans_model(model_name: str, save_dir: Path) -> None:
    """
    Download ProtTrans model to a specific directory

    Args:
        model_name: Model variant to download (e.g., "Rostlab/prot_t5_xl_uniref50")
        save_dir: Directory where to save the model
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} to {save_dir}...")
    print("This may take a while depending on model size and internet speed...")

    # Set cache directory for transformers
    os.environ['TRANSFORMERS_CACHE'] = str(save_dir)
    os.environ['HF_HOME'] = str(save_dir)

    # Download model and tokenizer
    model, tokenizer = load_protrans(model_name)

    # Save model and tokenizer to the specified directory
    model_save_path = save_dir / model_name.replace("/", "_")
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model downloaded and saved to: {model_save_path}")
    
    # Calculate directory size
    total_size = sum(f.stat().st_size for f in model_save_path.rglob('*') if f.is_file())
    print(f"Total size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download a ProtTrans model by name to a target directory.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Rostlab/prot_t5_xl_uniref50",
        choices=PROTRANS_MODELS,
        help="Name of the ProtTrans model variant from Hugging Face",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../embedding_model",
        help="Directory where the model will be downloaded",
    )
    args = parser.parse_args()

    download_protrans_model(args.model_name, Path(args.model_dir))


if __name__ == "__main__":
    main()
