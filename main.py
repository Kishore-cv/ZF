import os
import re
import cv2 as cv
from tqdm import tqdm
import sys

# Assuming the preprocess py files are in the 'Preprocess' folder in the current working directory
# If not, adjust the path accordingly
sys.path.append(os.path.join(os.getcwd(), 'Preprocess'))

# Import the extractor classes from the respective files
# Assuming each file defines its own tuned version of LhsToothExtractor or RhsToothExtractor
from parta_lhs import LhsToothExtractor as PartALhsExtractor
from parta_rhs import RhsToothExtractor as PartARhsExtractor
from partb_lhs import LhsToothExtractor as PartBLhsExtractor
from partb_rhs import RhsToothExtractor as PartBRhsExtractor

def prepare_dataset():
    # Define the day folders to process (only 31.1.2026 and 2.2.2026)
    days = [
        '/home/dell/ZF/ZF/31.1.2026',
        '/home/dell/ZF/ZF/2-2-2026/2.2.2026'
    ]

    # Output base directory
    output_base = '/home/dell/ZF/Dataset'
    os.makedirs(output_base, exist_ok=True)

    # Create output subfolders
    parts = ['PartA', 'PartB']
    sides = ['lhs', 'rhs']
    for part in parts:
        for side in sides:
            os.makedirs(os.path.join(output_base, part, side), exist_ok=True)

    # Counters for unique naming to avoid overwrites (since image names repeat across folders)
    counters = {
        'PartA_lhs': 0,
        'PartA_rhs': 0,
        'PartB_lhs': 0,
        'PartB_rhs': 0
    }

    total_saved = 0
    total_skipped = 0

    for day in days:
        if not os.path.exists(day):
            print(f"Warning: Day folder not found: {day}")
            continue

        # Find part folders (e.g., 1parta, 2partb, etc.)
        part_folders = [f for f in os.listdir(day) if os.path.isdir(os.path.join(day, f)) and re.search(r'\d*part[ab]', f.lower())]

        for part_folder in tqdm(part_folders, desc=f"Processing parts in {os.path.basename(day)}"):
            is_parta = 'parta' in part_folder.lower()
            part_type = 'PartA' if is_parta else 'PartB'

            # Select the appropriate extractors based on part type
            if is_parta:
                lhs_extractor = PartALhsExtractor()
                rhs_extractor = PartARhsExtractor()
            else:
                lhs_extractor = PartBLhsExtractor()
                rhs_extractor = PartBRhsExtractor()

            part_path = os.path.join(day, part_folder)

            # Find timestamp folders (e.g., 20260131_121930)
            ts_folders = [f for f in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, f)) and re.match(r'\d{8}_\d{6}', f)]

            for ts in ts_folders:
                ts_path = os.path.join(part_path, ts)

                # Find camera folders
                cam_folders = [f for f in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, f))]

                for cam in cam_folders:
                    if cam == '24678979':
                        side = 'lhs'
                        extractor = lhs_extractor
                    elif cam == '25320882':
                        side = 'rhs'
                        extractor = rhs_extractor
                    else:
                        continue  # Skip unknown cameras

                    cam_path = os.path.join(ts_path, cam)

                    # Find PNG images
                    images = sorted([f for f in os.listdir(cam_path) if f.endswith('.png')])

                    for img_name in tqdm(images, desc=f"Processing {side} images in {ts}", leave=False):
                        img_path = os.path.join(cam_path, img_name)
                        img = cv.imread(img_path)
                        if img is None:
                            total_skipped += 1
                            continue

                        # Extract the cropped region
                        cropped = extractor.extract(img)
                        if cropped.size == 0:
                            total_skipped += 1
                            continue

                        # Save with unique name using counter
                        key = f'{part_type}_{side}'
                        counters[key] += 1
                        save_name = f'{counters[key]:06d}.png'
                        save_dir = os.path.join(output_base, part_type, side)
                        save_path = os.path.join(save_dir, save_name)

                        cv.imwrite(save_path, cropped)
                        total_saved += 1

    print("\nDataset preparation completed.")
    print(f"Output root: {output_base}")
    print(f"Total images saved: {total_saved}")
    print(f"Total images skipped: {total_skipped}")

if __name__ == "__main__":
    prepare_dataset()