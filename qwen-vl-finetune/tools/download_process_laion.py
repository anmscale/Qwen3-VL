import argparse
import json
import os
from pathlib import Path
import requests
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = "Write a caption for the following image: <image>."


def download_image(url, output_path):
    """Download an image from a URL and save it to the specified path.
    
    Args:
        url: The URL of the image to download
        output_path: The path where the image should be saved
        
    Returns:
        The output_path if successful, None if failed
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download the image
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Save the image
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def save_jsonl(data, output_path):
    """Save a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        output_path: Path to the output JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(data)} samples to {output_path}")


def main(min_width, max_width, min_height, max_height, dry_run=False, output_path=None, max_images=-1, overwrite=False, val_percent=0.0):
    # Check if output directory exists and is not empty
    if output_path and os.path.exists(output_path):
        if os.listdir(output_path):  # Directory is not empty
            if not overwrite:
                raise ValueError(
                    f"Output directory '{output_path}' is not empty. "
                    "Use --overwrite to overwrite existing files."
                )
            else:
                print(f"Warning: Output directory '{output_path}' is not empty. Overwriting enabled.")
    
    try:
        ds = load_dataset("laion/relaion-pop")
    except Exception as e:
        print(e)
        print("Please log in using the huggingface-cli login command.")
        print("huggingface-cli login")
        return

    print("Loaded Dataset")
    ds = ds['train']
    print("Length: ", len(ds))
    
    # Statistics tracking
    total_processed = 0
    filtered_by_size = 0
    images_downloaded = 0
    failed_downloads = 0
    cogvlm_captions = 0
    llava_captions = 0
    
    output_jsons = []
    
    for i in tqdm(ds):
        total_processed += 1
        
        # Check if we've reached the max number of images
        if max_images != -1 and images_downloaded >= max_images:
            print(f"Reached maximum number of images ({max_images}). Stopping.")
            break
        
        height = i['height']
        width = i['width']
        
        # Filter by size
        if height < min_height or height > max_height or width < min_width or width > max_width:
            filtered_by_size += 1
            continue
        
        url = i['url']
        cogvlm_caption = i['cogvlm_caption']
        llava_caption = i['llava_caption']
        img_path = f"{output_path}/images/{i['key']}.jpg"
        
        if not dry_run:
            downloaded_path = download_image(url, img_path)
            if downloaded_path is None:
                failed_downloads += 1
                continue
            img_path = downloaded_path
        images_downloaded += 1
        
        # Add cogvlm caption if available
        if len(cogvlm_caption) > 0:
            output_jsons.append({
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n" + SYSTEM_PROMPT,
                    },
                    {
                        "from": "gpt",
                        "value": cogvlm_caption,
                    },
                ]
            })
            cogvlm_captions += 1
        
        # Add llava caption if available
        if len(llava_caption) > 0:
            output_jsons.append({
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n" + SYSTEM_PROMPT,
                    },
                    {
                        "from": "gpt",
                        "value": llava_caption,
                    },
                ]
            })
            llava_captions += 1

    # Split data into train and validation sets
    total_samples = len(output_jsons)
    if val_percent > 0:
        val_split_idx = int(total_samples * (1 - val_percent))
        train_jsons = output_jsons[:val_split_idx]
        val_jsons = output_jsons[val_split_idx:]
        
        save_jsonl(train_jsons, f"{output_path}/laion_pop_train.jsonl")
        save_jsonl(val_jsons, f"{output_path}/laion_pop_val.jsonl")
    else:
        # No validation split, save all as training data
        save_jsonl(output_jsons, f"{output_path}/laion_pop_train.jsonl")
        train_jsons = output_jsons
        val_jsons = []
    
    # Print statistics
    print("\n" + "="*50)
    print("DOWNLOAD STATISTICS")
    print("="*50)
    print(f"Total samples processed: {total_processed}")
    print(f"Filtered by size: {filtered_by_size}")
    print(f"Images downloaded: {images_downloaded}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"CogVLM captions: {cogvlm_captions}")
    print(f"Llava captions: {llava_captions}")
    print(f"Total conversation pairs: {len(output_jsons)}")
    print(f"Training samples: {len(train_jsons)}")
    print(f"Validation samples: {len(val_jsons)}")
    if val_percent > 0:
        print(f"Validation split: {val_percent*100:.1f}%")
    print("="*50)

    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process LAION dataset")
    parser.add_argument("--output-dir", type=str, dest="output_dir", help="Output directory path")
    parser.add_argument("--min-width", type=int, default=1024, help="Minimum image width")
    parser.add_argument("--max-width", type=int, default=10000, help="Maximum image width")
    parser.add_argument("--min-height", type=int, default=1024, help="Minimum image height")
    parser.add_argument("--max-height", type=int, default=10000, help="Maximum image height")
    parser.add_argument("--dry-run", action="store_true", help="Run without downloading images")
    parser.add_argument("--max-images", type=int, default=-1, help="Maximum number of images to download (-1 for no limit)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting files in non-empty output directory")
    parser.add_argument("--val-percent", type=float, default=0.20, dest="val_percent", help="Validation set percentage (0.0-1.0, e.g., 0.1 for 10%%)")
    
    args = parser.parse_args()
    
    # Validate val_percent
    if not 0.0 <= args.val_percent <= 1.0:
        parser.error("--val-percent must be between 0.0 and 1.0")
    
    main(
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        dry_run=args.dry_run,
        output_path=args.output_dir,
        max_images=args.max_images,
        overwrite=args.overwrite,
        val_percent=args.val_percent
    )