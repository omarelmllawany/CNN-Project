#!/usr/bin/env python
"""
Simple automated test to verify `SatelliteClassifier.predict` and main script run.
"""
import os
import sys
from PIL import Image
import numpy as np

# Ensure project root is importable when running from tests/ directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def create_dummy_image(path=None, size=(64,64)):
    if path is None:
        path = os.path.join(ROOT, 'test_rand.png')
    arr = (np.random.rand(size[0], size[1], 3) * 255).astype('uint8')
    img = Image.fromarray(arr)
    img.save(path)
    return path

def run_predict_test(image_path='test_rand.png'):
    from main_2 import SatelliteClassifier

    clf = SatelliteClassifier('cnn')
    result, _ = clf.predict(image_path)

    # Basic structure checks
    assert isinstance(result, dict), 'Result must be a dict'
    assert 'class_index' in result and isinstance(result['class_index'], int)
    assert 'class_name' in result and isinstance(result['class_name'], str)
    assert 'confidence' in result and isinstance(result['confidence'], float)
    assert 'probabilities' in result and isinstance(result['probabilities'], list)
    assert len(result['probabilities']) == 10

    print('run_predict_test: PASS')

def main():
    img_path = create_dummy_image()
    run_predict_test(img_path)

    # Run the main script in predict mode and save the visualization (no GUI)
    save_output = os.path.join(ROOT, 'test_visualization.png')
    cmd = f'python main_2.py --mode predict --model cnn --image "{img_path}" --save_path "{save_output}" --no_show'
    print('Running main script:', cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit(f'main_2.py exited with code {rc}')

if __name__ == '__main__':
    main()
