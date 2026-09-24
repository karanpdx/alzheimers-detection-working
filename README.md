# Alzheimer's Detection from MRI

A NeuroTech@Berkeley (Sp26) project classifying Alzheimer's disease stage from brain MRI scans, and visualizing *what the model is actually looking at* when it makes that call.

## What it does

Fine-tunes [MedViT](https://github.com/Omid-Nejati/MedViTV2) — a medical-imaging vision transformer — to classify MRI scans into four stages: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`. Model architecture and training code are from the MedViT paper (Manzari et al., [arXiv:2502.13693](https://arxiv.org/abs/2502.13693)); `MedViTV2-main/` is the upstream library, vendored here for training.

## My role: interpretability

I built the [Grad-CAM visualization notebook](MedViTV2-main/Tutorials/Visualization_Alzheimers.ipynb) that inspects the fine-tuned model's predictions class-by-class. For a given MRI scan, it overlays a Grad-CAM heatmap on the model's final attention layer, showing which regions of the brain it weighted most heavily to reach a diagnosis — alongside the predicted class and confidence. This is what interpretability work looks like for a clinical-adjacent model: it's not enough for the classifier to be accurate, you need to see whether it's attending to plausible anatomy rather than to scan artifacts.

## Stack

PyTorch · MedViT (KAN-integrated medical vision transformer) · `pytorch-grad-cam` · torchvision

## Repo layout

```
MedViTV2-main/                          upstream MedViT library (training + model code)
  Tutorials/Visualization_Alzheimers.ipynb   Grad-CAM interpretability notebook (my contribution)
  checkpoints/                          per-dataset config references
```
