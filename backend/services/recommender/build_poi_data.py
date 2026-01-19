import os
import logging
from backend.services.recommender.downsampling import build_filtered_poi_subset
from backend.services.recommender.fsq_fetch import download_all_enriched
from backend.services.recommender.features import save_venue_features
from backend.services.recommender.embeddings import save_venue_embeddings
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()


def build_poi_data_pipeline():
    raw_path = "data/recommender/raw"
    processed_path = "data/recommender/processed"
    final_output_files = [
        os.path.join(processed_path, "venue_features.parquet"),
        os.path.join(processed_path, "venue_embeddings.parquet"),
    ]
    if all(os.path.exists(f) for f in final_output_files):
        logging.info("Processed POI data already exists. Skipping build process.")
        return  # Esce dalla funzione se i file esistono già
    # --- FINE DEL CONTROLLO ---

    logging.info("Processed POI data not found. Starting the data build pipeline.")
    logging.warning("This process may take a long time and will only run once.")

    # Controlla che i dati grezzi necessari siano presenti prima di iniziare
    raw_input_files = [
        os.path.join(raw_path, "dataset_TIST2015_Checkins.txt"),
        os.path.join(raw_path, "dataset_TIST2015_POIs.txt"),
    ]
    if not all(os.path.exists(f) for f in raw_input_files):
        logging.error(
            "Raw data files not found in 'data/recommender/raw/'. Cannot build POI data."
        )
        logging.error(
            "Please download the TIST2015 dataset and place the files in the correct folder."
        )
        raise FileNotFoundError("Raw TIST2015 dataset not found. Aborting.")

    # Crea le cartelle se non esistono
    os.makedirs(processed_path, exist_ok=True)

    print("Step 1: Downsampling POIs...")
    venues_to_fetch_df = build_filtered_poi_subset(
        checkins_path=os.path.join(raw_path, "dataset_TIST2015_Checkins.txt"),
        poi_path=os.path.join(raw_path, "dataset_TIST2015_POIs.txt"),
        keep_fraction=0.05,
    )
    venues_to_fetch_df.to_parquet(
        os.path.join(processed_path, "venues_to_fetch.parquet")
    )

    print("Step 2: Fetching details from Foursquare API...")
    download_all_enriched(
        venues_to_fetch_df, os.path.join(processed_path, "venue_details.jsonl")
    )

    print("Step 3: Building feature frame...")
    features_df = save_venue_features(
        os.path.join(processed_path, "venue_details.jsonl"),
        os.path.join(processed_path, "venue_features.parquet"),
    )

    print("Step 4: Generating POI embeddings...")
    save_venue_embeddings(
        features_df, os.path.join(processed_path, "venue_embeddings.parquet")
    )

    print("POI data pipeline completed successfully!")


if __name__ == "__main__":
    build_poi_data_pipeline()
