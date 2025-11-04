import argparse
import esm
import os
from pathlib import Path
import torch


# Valid ESM-2 names (UR50D training)
ESM2_MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]

def load_esm2(model_name: str):
    """
    Tries, in order:
      1) esm.pretrained.<name>()  [newer fair-esm]
      2) esm.pretrained.load_model_and_alphabet_hub(name)  [hub helper]
      3) esm.pretrained.load_model_and_alphabet(name)  [older API]
    """

    # 1) direct constructor if present
    if hasattr(esm, "pretrained") and hasattr(esm.pretrained, model_name):
        return getattr(esm.pretrained, model_name)()
    # 2) hub helper (newer)
    if hasattr(esm.pretrained, "load_model_and_alphabet_hub"):
        return esm.pretrained.load_model_and_alphabet_hub(model_name)
    # 3) legacy helper (older)
    if hasattr(esm.pretrained, "load_model_and_alphabet"):
        return esm.pretrained.load_model_and_alphabet(model_name)
    raise RuntimeError(
        f"Cannot load '{model_name}'. Your installed 'esm' lacks known loaders."
    )


def download_esm2_model(model_name: str, save_dir: Path) -> None:
    """
    Download ESM-2 model to a specific directory

    Args:
        model_name: Model variant to download
        save_dir: Directory where to save the model
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} to {save_dir}...")
    print("This may take a while depending on model size and internet speed...")

    # Set cache directory before loading model
    os.environ['TORCH_HOME'] = str(save_dir)

    # Download model
    model, alphabet = load_esm2(model_name)

    # Save model weights
    model_path = save_dir / f"{model_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
    }, model_path)

    print(f"Model downloaded and saved to: {model_path}")
    print(f"File size: {model_path.stat().st_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download an ESM-2 model by name to a target directory.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D",
        choices=ESM2_MODELS,
        help="Name of the ESM-2 model variant",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../embedding_model",
        help="Directory where the model will be downloaded",
    )
    args = parser.parse_args()

    download_esm2_model(args.model_name, Path(args.model_dir))


if __name__ == "__main__":
    main()
