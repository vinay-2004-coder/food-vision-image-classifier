# Food Vision — Food-101 Image Classification (TensorFlow)

## Overview
End-to-end image classification on the Food-101 dataset using TensorFlow/Keras and transfer learning.

## Dataset
- Food-101 (via TensorFlow Datasets)
- 101 classes
- ~75,750 training images, ~25,250 test images

## Approach
- Transfer learning with a pretrained CNN backbone
- Efficient `tf.data` pipeline (batching/prefetching)
- Mixed precision training (if supported)
- Optional fine-tuning

## How to Run
### Option A: Google Colab (easiest)
1. Upload `Food_Vision.ipynb` and `helper_functions.py`
2. Run the notebook top-to-bottom

### Option B: Local
```bash
pip install -r requirements.txt
jupyter notebook
