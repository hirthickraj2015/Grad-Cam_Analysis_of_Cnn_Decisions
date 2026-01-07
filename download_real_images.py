#!/usr/bin/env python3
# Script to download test images from wikimedia commons

import urllib.request
import ssl
from pathlib import Path
import sys

# chest X-ray URLs (public domain)
SAMPLE_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Chest_radiograph_in_influenza_and_Haemophilus_influenzae%2C_posteroanterior%2C_annotated.jpg/800px-Chest_radiograph_in_influenza_and_Haemophilus_influenzae%2C_posteroanterior%2C_annotated.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Chest_X-ray_in_influenza_A_H1N1.png/800px-Chest_X-ray_in_influenza_A_H1N1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Aortic_rupture_chest_x-ray.jpg/800px-Aortic_rupture_chest_x-ray.jpg",
]

# ImageNet sample images for testing
IMAGENET_SAMPLES = [
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
    output_dir = Path('input_images')
    output_dir.mkdir(exist_ok=True)

    urls = IMAGENET_SAMPLES if use_imagenet else SAMPLE_IMAGES
    prefix = "real" if use_imagenet else "xray"

    print(f"Downloading {len(urls)} {'ImageNet' if use_imagenet else 'medical'} images...")
    print(f"Output directory: {output_dir.absolute()}\n")

    # disable SSL verification
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

    print(f"\nDownloaded {downloaded}/{len(urls)} images")
    print(f"Location: {output_dir.absolute()}")

    if downloaded == 0:
        print("\nNo images downloaded. Check internet connection.")
        return False

    return True

if __name__ == "__main__":
    use_imagenet = "--imagenet" in sys.argv or "--real" in sys.argv
    success = download_images(use_imagenet=use_imagenet)

    if success:
        print("\nNext steps:")
        print("  Run the notebooks in order:")
        print("  - 1_create_dataset.ipynb")
        print("  - 2_gradcam.ipynb")
        print("  - 3_layercam.ipynb")
        print("  - 4_hybrid_gradcam_layercam.ipynb")
        print("  - 5_evaluation_comparison.ipynb")
    else:
        sys.exit(1)
