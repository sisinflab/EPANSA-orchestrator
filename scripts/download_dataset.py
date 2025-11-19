from huggingface_hub import snapshot_download
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset_directory(repo_id: str, repo_type: str, local_dir: str) -> None:
    """
    Downloads an entire directory from a Hugging Face Hub repository.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        repo_type (str): The type of repository (e.g., "dataset").
        local_dir (str): The local directory where files will be saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    logger.info(f"Downloading entire repository {repo_id} to {local_dir}")
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=[".gitattributes", "README.md"]
    )
    logger.info(f"Successfully downloaded {repo_id} to {local_dir}")
    
    # List downloaded files for verification
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            logger.info(f"  Downloaded: {os.path.relpath(file_path, local_dir)} ({file_size} bytes)")
    
if __name__ == "__main__":
    repo_id = "GabrieleConte/ForsquareDataset"
    local_dir = "data/"
    
    # Check if data already exists
    if os.path.exists(os.path.join(local_dir, "recommender/processed/venue_features.parquet")) and \
       os.path.exists(os.path.join(local_dir, "recommender/processed/venue_embeddings.parquet")) and \
       os.path.exists(os.path.join(local_dir, "recommender/raw/dataset_TIST2015_Checkins.txt")):
        logger.info("Dataset already exists. Skipping download.")
    else:
        download_dataset_directory(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir
        )