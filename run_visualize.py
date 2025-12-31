"""Helper to run visualization in headless mode and save output.

Usage:
    python run_visualize.py --image path\to\image.jpg --model cnn --model_path path\to\weights.pth --save_path out.png
"""
import argparse
import os
from main_2 import SatelliteClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='cnn', choices=['cnn','vit'])
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--save_path', default='visualization.png')

    args = parser.parse_args()

    clf = SatelliteClassifier(args.model, args.model_path)
    res = clf.visualize_prediction(args.image, save_path=args.save_path, show=False)
    if res:
        print(f"Saved visualization to: {args.save_path}")

if __name__ == '__main__':
    main()
