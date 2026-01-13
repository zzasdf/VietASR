import json
import os
import argparse


def download_model(save_dir: str, model_name: str, no_download: bool = False) -> None:
    """
    Download ASR model files based on the specified model name.

    Args:
        save_dir (str): Directory to save the downloaded model files
        model_name (str): Name or index of the model to download
        no_download (bool): If True, only show available models without downloading

    Raises:
        AssertionError: If model_name is not provided
    """
    # Load model information from JSON file
    models = json.load(open('asr_models.json'))

    # Display available models
    num_model = len(models.keys())
    print("====AVAILABLE MODELs====")
    for i, m_name in enumerate(models):
        print("=======================")
        print(f"{i+1}/{num_model}: {m_name}")
        m = models[m_name]
        for k, v in m.items():
            if isinstance(v, str):
                print(f" + {k}:{v}")
    print("=======================")

    # Validate model_name parameter
    assert model_name is not None and model_name != "", "model_name must be set"

    # Handle numeric model selection
    if isinstance(model_name, int) or model_name.isdigit():
        model_name = list(models.keys())[int(model_name) - 1]

    # Verify model exists
    if model_name not in models.keys():
        for model in list(models.keys()):
            print(model)
        exit()

    if not no_download:
        print(f"Downloading model: {model_name}")
        model_files = models[model_name]["model"]
        save_dir = os.path.join(save_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)

        # Download each model file
        for k, v in model_files.items():
            if v is not None and v != "":
                print(f" + downloading: {k}")
                file_path = f"{save_dir}/{k}"
                if os.path.exists(file_path):
                    print(f"file: {file_path} already exists")
                else:
                    os.system(f"wget -q --show-progress --progress=bar:force {v} -O {file_path}")
    print("Download process completed")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='ASR Model Downloader - Download Automatic Speech Recognition models'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="model_file",
        help='Directory to save downloaded model files'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        help='Name or index of the model to download'
    )
    parser.add_argument(
        '--no_download',
        type=bool,
        default=False,
        help='If True, only show model list without downloading'
    )

    # Parse arguments and execute download
    args = parser.parse_args()
    download_model(
        save_dir=args.save_dir,
        model_name=args.model_name,
        no_download=args.no_download
    )
