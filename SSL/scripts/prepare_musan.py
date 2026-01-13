import logging
from pathlib import Path
from lhotse.recipes import prepare_musan

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Define paths
    corpus_dir = Path("/vietasr/data3/musan") # Using docker path structure
    output_dir = Path("data/wav")
    
    # Create directories if they don't exist
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading and preparing MUSAN dataset in {corpus_dir}")
    
    # Download and prepare MUSAN
    # This will download the dataset if it's not present
    musan_manifests = prepare_musan(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        parts=("music", "speech", "noise")
    )
    
    # Combine all cuts into a single manifest
    from lhotse import CutSet
    from lhotse.manipulation import combine
    
    all_cuts = []
    for part in ["music", "speech", "noise"]:
        cuts_file = output_dir / f"musan_cuts_{part}.jsonl.gz"
        if cuts_file.exists():
            logging.info(f"Loading existing cuts for {part}")
            cuts = CutSet.from_file(cuts_file)
            all_cuts.append(cuts)
        else:
            logging.info(f"Cuts for {part} not found, skipping")
    
    if all_cuts:
        combined_cuts = combine(all_cuts)
        combined_cuts.to_file(output_dir / "musan_cuts.jsonl.gz")
        logging.info(f"Combined MUSAN cuts saved to {output_dir / 'musan_cuts.jsonl.gz'}")
    else:
        logging.error("No cuts found to combine!")
    
    logging.info(f"MUSAN preparation completed. Manifests saved to {output_dir}")

if __name__ == "__main__":
    main()
