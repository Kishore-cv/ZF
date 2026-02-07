import os
import re
import cv2 as cv
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.getcwd(), 'Preprocess'))

from parta_lhs import LhsToothExtractor as PartALhsExtractor
from parta_rhs import RhsToothExtractor as PartARhsExtractor
from partb_lhs import LhsToothExtractor as PartBLhsExtractor
from partb_rhs import RhsToothExtractor as PartBRhsExtractor

def prepare_rhs_only():
    days = [
        '/home/dell/ZF/ZF/31.1.2026',
        '/home/dell/ZF/ZF/2-2-2026/2.2.2026'
    ]
 
    output_base = '/home/dell/ZF/Dataset'
    os.makedirs(output_base, exist_ok=True)

    parts = ['PartA', 'PartB']
    for part in parts:
        os.makedirs(os.path.join(output_base, part, 'rhs'), exist_ok=True)

    counters = {
        'PartA_rhs': 0,
        'PartB_rhs': 0
    }

    total_saved = 0
    total_skipped = 0

    for day in days:
        if not os.path.exists(day):
            print(f"Warning: Day folder not found: {day}")
            continue

        part_folders = [f for f in os.listdir(day) if os.path.isdir(os.path.join(day, f)) and re.search(r'\d*part[ab]', f.lower())]

        for part_folder in tqdm(part_folders, desc=f"Processing parts in {os.path.basename(day)}"):
            is_parta = 'parta' in part_folder.lower()
            part_type = 'PartA' if is_parta else 'PartB'

            if is_parta:
                rhs_extractor = PartARhsExtractor()
            else:
                rhs_extractor = PartBRhsExtractor()

            part_path = os.path.join(day, part_folder)

            ts_folders = [f for f in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, f)) and re.match(r'\d{8}_\d{6}', f)]

            for ts in ts_folders:
                ts_path = os.path.join(part_path, ts)

                cam_folders = [f for f in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, f))]

                for cam in cam_folders:
                    if cam != '25320882':          # ← ONLY RHS
                        continue

                    side = 'rhs'
                    extractor = rhs_extractor

                    cam_path = os.path.join(ts_path, cam)

                    images = sorted([f for f in os.listdir(cam_path) if f.endswith('.png')])

                    for img_name in tqdm(images, desc=f"Processing RHS images in {ts}", leave=False):
                        img_path = os.path.join(cam_path, img_name)
                        img = cv.imread(img_path)
                        if img is None:
                            total_skipped += 1
                            continue

                        cropped = extractor.extract(img)
                        if cropped.size == 0:
                            total_skipped += 1
                            continue

                        key = f'{part_type}_rhs'
                        counters[key] += 1
                        save_name = f'{counters[key]:06d}.png'
                        save_dir = os.path.join(output_base, part_type, 'rhs')
                        save_path = os.path.join(save_dir, save_name)

                        cv.imwrite(save_path, cropped)
                        total_saved += 1

                        # Optional: print progress every 50 images
                        if total_saved % 50 == 0:
                            print(f"Saved {total_saved} RHS images so far...")

    print("\nRHS-only reprocessing completed.")
    print(f"Output folders: {output_base}/PartA/rhs  and  {output_base}/PartB/rhs")
    print(f"Total RHS images saved: {total_saved}")
    print(f"Total skipped: {total_skipped}")

if __name__ == "__main__":
    prepare_rhs_only()