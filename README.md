# Computer Vision — Transfer Learning with MobileNetV2 (Portfolio Sample)

This repository presents a **clean transfer-learning workflow** using **MobileNetV2** in Keras/TensorFlow.
It trains a small classifier on a toy dataset (CIFAR-10 resized to 224×224) to keep the example **fully reproducible**
and includes a **simple inference demo** on a sample image.

## What this shows
- **Transfer learning**: MobileNetV2 (frozen backbone) + custom classification head
- **Data pipeline**: resize & normalize images; train/val split; on-the-fly augmentation
- **Evaluation**: accuracy curves, confusion matrix, and a brief error analysis
- **Inference demo**: run the trained model on a sample image and visualize the prediction

> In real projects, replace CIFAR-10 with your dataset (medical or natural images).
> The notebook is structured so you can plug in a folder-based dataset easily.

## Repo structure
```
.
├── notebooks
│   └── cv_transfer_learning.ipynb   # end-to-end workflow
├── assets
│   └── sample.jpg                   # small placeholder image for inference demo
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/cv_transfer_learning.ipynb
```

## Notes
- The notebook uses **CIFAR-10** by default (downloads via Keras). For your own data, follow the cell that shows how to use
  a directory with `train/` and `val/` subfolders and class-named directories (standard Keras flow).
- Model weights are **not committed**; see the checkpoint cell for saving/loading if needed.

## License
MIT
