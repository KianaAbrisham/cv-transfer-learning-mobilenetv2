# MobileNetV2 Transfer Learning on CIFAR-10

**Author:** [Kiana Pilevar Abrisham](https://github.com/KianaAbrisham)

A TensorFlow/Keras image-classification experiment with frozen ImageNet features,
validation-based checkpoint selection, held-out evaluation, and saved-model inference.

**[Open the executed notebook](notebooks/cv_transfer_learning.ipynb)** to inspect
training curves, a confusion matrix, error examples, and checkpoint checks.

## What this project demonstrates

- Memory-conscious input handling: native 32 × 32 images are batched before
  resizing inside the model. The complete dataset is never expanded to 224 × 224.
- A frozen MobileNetV2 backbone with a trainable classification head.
- Training-only augmentation, stratified train/validation splits, and a separate
  official test partition.
- Checkpoint selection by validation loss, followed by test accuracy, macro F1,
  a majority-class baseline, per-class diagnostics, and error inspection.
- Embedded resizing and normalization in the saved `.keras` model, with verified
  checkpoint reload and prediction from an image file.
- Saved configuration, package versions, class names, selected indices, training
  history, predictions, metrics, and plots.

## Recorded quick-run results

Checked on **2026-09-27**, using Python **3.12.14**,
TensorFlow **2.20.0**, and a Linux CPU runtime.
All ten code cells executed successfully in one IPython session.

| Item | Measured result |
| --- | --- |
| Training / validation / test images | 2,000 / 500 / 1,000 |
| Epochs completed | 3 |
| Selected epoch, by validation loss | 3 |
| Test accuracy | 77.0% |
| Test macro F1 | 0.770 |
| Majority-class baseline accuracy | 10.0% |
| Checkpoint reload | Matching predictions for eight held-out images |
| Image-file inference | Passed |

These are measured results from a fixed, stratified subset, **not a full CIFAR-10
benchmark**. Test images did not select the checkpoint. The notebook contains
the actual cell outputs, plots, and experiment metadata.

## Setup

Use **Python 3.12** and a separate environment. From the repository's main folder:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mobilenet-portfolio --display-name "Python (MobileNet)"
python -m notebook notebooks/cv_transfer_learning.ipynb
```

Select **Python (MobileNet)** as the kernel and use **Restart Kernel and Run All**.
The pinned requirements target CPU execution; recorded validation was on Linux.
The first run downloads CIFAR-10 (about 170 MB) and ImageNet weights (about 9.4 MB)
to Keras's cache.

## Run modes

| Setting in the configuration cell | Train / validation / test | Epoch limit |
| --- | --- | --- |
| `MODE = "quick"` — default | 2,000 / 500 / 1,000 | 3 |
| `MODE = "full"` | 40,000 / 10,000 / 10,000 | 15 |

Full mode takes more CPU time and has not been validated in this revision.
With batch size 16, one resized float32 input batch is about **9.6 MB**. This
excludes weights, activations, dataset storage, and other runtime memory; it is
not a total RAM requirement.

Each execution creates a new directory beneath `runs/`. Keep generated datasets
and checkpoints outside Git history. The model accepts RGB pixels in **[0, 255]**;
resizing and normalization are embedded in its graph. The notebook's
`predict_image(path)` function demonstrates image-file inference.

## Scope and limitations

- Only the classification head is trained; backbone fine-tuning is not implemented.
- Resizing does not add detail to CIFAR-10's original 32 × 32 pixels.
- Softmax scores are not calibrated confidence estimates.
- Fixed seeds and saved indices help repeatability. Hardware/library differences
  can affect timings and exact numerical results.
- Test-error inspection is a diagnostic; model changes based on those errors
  require a new independent evaluation.
- Custom folder datasets require their own loader, class mapping, and splits;
  this is not an implemented workflow in the notebook.

## References

- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow transfer-learning guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras MobileNetV2 API](https://keras.io/api/applications/mobilenet/)

## Development context

This portfolio experiment was revised with AI coding assistance for the input pipeline, checkpoint checks, evaluation, notebook execution, and documentation. It uses Keras's pretrained MobileNetV2 implementation and official CIFAR-10 data. The recorded three-epoch subset run is the evidence for this version; a larger benchmark remains a separate experiment.

Recorded checks were run in a hosted Linux CPU environment. See the [portfolio development notes](https://github.com/KianaAbrisham/KianaAbrisham/blob/main/docs/DEVELOPMENT.md) for execution provenance and the scope of AI assistance.

## License

Repository code uses the existing [MIT license](LICENSE). Dataset and pretrained
weight terms remain with their respective providers.
