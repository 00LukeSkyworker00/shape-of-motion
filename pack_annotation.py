import cv2
import os
import numpy as np
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some images.")

# Add arguments
parser.add_argument("--data_dir", type=str, help="Path to the dataset directory.")
parser.add_argument("--out_dir", type=str, help="Path to the output directory.")

# Parse arguments
args = parser.parse_args()

ano_dir = os.path.join(args.data_dir,'ano')
ano = [f for f in os.listdir(ano_dir) if f.endswith('.png')]
ano = sorted(ano)
annotations = []
for png in ano:
    ano_path = os.path.join(ano_dir, png)
    png = cv2.imread(ano_path)
    annotations.append(png[:,:,0:1])

annotations = np.stack(annotations, axis=0)

np.savez_compressed(os.path.join(args.out_dir, 'ground_truth.npz'), ano=annotations)