#!/usr/bin/env python3
"""
Download Real Medical Images for Testing
=========================================
Downloads sample chest X-ray images from a public dataset.
"""

import urllib.request
import ssl
from pathlib import Path
import sys

# Sample chest X-ray URLs from NIH dataset (publicly available)
SAMPLE_IMAGES = [
    # These are public domain chest X-rays
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Chest_radiograph_in_influenza_and_Haemophilus_influenzae%2C_posteroanterior%2C_annotated.jpg/800px-Chest_radiograph_in_influenza_and_Haemophilus_influenzae%2C_posteroanterior%2C_annotated.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Chest_X-ray_in_influenza_A_H1N1.png/800px-Chest_X-ray_in_influenza_A_H1N1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Aortic_rupture_chest_x-ray.jpg/800px-Aortic_rupture_chest_x-ray.jpg",
]

# Alternative: Download ImageNet sample images (diverse, well-structured)
IMAGENET_SAMPLES = [
    # Various ImageNet categories - diverse real images
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/800px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/YellowLabradorLooking_new.jpg/800px-YellowLabradorLooking_new.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Phoenicopterus_ruber_in_S%C3%A3o_Paulo_Zoo.jpg/800px-Phoenicopterus_ruber_in_S%C3%A3o_Paulo_Zoo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/800px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Colosseum_in_Rome%2C_Italy_-_April_2007.jpg/800px-Colosseum_in_Rome%2C_Italy_-_April_2007.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Golden_Retriever_standing_Tucker.jpg/800px-Golden_Retriever_standing_Tucker.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/African_Bush_Elephant.jpg/800px-African_Bush_Elephant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant_Digitally_Enhanced.jpg/800px-African_Bush_Elephant_Digitally_Enhanced.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/800px-American_Eskimo_Dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Red_Kitten_01.jpg/800px-Red_Kitten_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Stonehenge.jpg/800px-Stonehenge.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Giant_Panda_Tai_Shan.JPG/800px-Giant_Panda_Tai_Shan.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/800px-Cute_dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/The_Great_Wall_of_China_at_Jinshanling-edit.jpg/800px-The_Great_Wall_of_China_at_Jinshanling-edit.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Eiffel_Tower_at_night%2C_Paris_20_May_2023.jpg/600px-Eiffel_Tower_at_night%2C_Paris_20_May_2023.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg/800px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Sezierpraeparat-Weichteile_eines_Frosches-043.jpg/800px-Sezierpraeparat-Weichteile_eines_Frosches-043.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Socks-clinton.jpg/800px-Socks-clinton.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Lynx_lynx-4.JPG/800px-Lynx_lynx-4.JPG",
]

def download_images(use_imagenet=True):
    """Download sample images for testing."""
    output_dir = Path('medical_images')
    output_dir.mkdir(exist_ok=True)

    # Use ImageNet images by default (better for ResNet testing)
    urls = IMAGENET_SAMPLES if use_imagenet else SAMPLE_IMAGES
    prefix = "real" if use_imagenet else "xray"

    print(f"Downloading {len(urls)} {'ImageNet' if use_imagenet else 'medical'} images...")
    print(f"Output directory: {output_dir.absolute()}\n")

    # Disable SSL verification for downloads (some certificates may be old)
    ssl._create_default_https_context = ssl._create_unverified_context

    downloaded = 0
    for idx, url in enumerate(urls, 1):
        try:
            filename = f"{prefix}_{idx:03d}.jpg"
            filepath = output_dir / filename

            print(f"[{idx}/{len(urls)}] Downloading {filename}...", end=' ')
            urllib.request.urlretrieve(url, filepath)
            print("✓")
            downloaded += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"Downloaded {downloaded}/{len(urls)} images successfully")
    print(f"Location: {output_dir.absolute()}")
    print('='*70)

    if downloaded == 0:
        print("\n⚠ No images downloaded. Check your internet connection.")
        return False

    return True

if __name__ == "__main__":
    use_imagenet = "--imagenet" in sys.argv or "--real" in sys.argv
    success = download_images(use_imagenet=use_imagenet)

    if success:
        print("\nNext steps:")
        print("  1. Run: jupyter notebook 3_run_batch_experiments.ipynb")
        print("  2. Or run: jupyter notebook 4_complete_pipeline.ipynb")
        print("\nThese real images will produce meaningful, diverse results!")
    else:
        sys.exit(1)
