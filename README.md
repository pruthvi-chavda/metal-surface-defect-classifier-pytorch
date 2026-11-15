# Industrial Metal Surface Defect Classifier

An end-to-end deep learning pipeline for real-time detection of micro-textures and defects (e.g., oil spots, dents, scratches) on industrial metal surfaces. 

This project demonstrates the progressive evolution of a computer vision architecture—starting from a custom 5-layer CNN baseline and scaling to a deep-layer fine-tuned EfficientNet-B0—optimized for low-VRAM hardware using Automatic Mixed Precision (AMP) and deployed via a FastAPI/Streamlit microservice architecture.

## 🧠 Architectural Evolution & Strategy

Industrial surface defects present a unique challenge: micro-textures like "oil spots" and "dents" often share identical low-level generic edges (shadows/dark patches) causing severe class collision in standard pre-trained models. 

This repository documents the structural pivot required to solve this:

1. **Phase 1: Custom Baseline CNN (`base_cnn.py`)**
   - Built a 5-layer CNN with `GroupNorm` and `AdaptiveAvgPool2d`.
   - Addressed severe class imbalance using PyTorch's `WeightedRandomSampler` based on Inverse Class Frequency.
   - Result: Struggled to generalize across complex textures.

2. **Phase 2: Transfer Learning (Frozen Backbone)**
   - Initialized `ResNet18` and `EfficientNet-B0` using ImageNet weights (`IMAGENET1K_V1`).
   - Adapted 1-channel grayscale industrial data to 3-channel architecture using `transforms.Grayscale(num_output_channels=3)`.
   - Result: Model hit a performance ceiling. Frozen ImageNet weights (trained on macro objects like dogs/cars) failed to separate industrial micro-textures.

3. **Phase 3: Deep-Layer Fine-Tuning (`pretrained_models.py`)**
   - **The Solution:** Implemented `MetalDefectEfficientNetUnfreezed`.
   - Froze the early network layers but intentionally **unfroze the final 3 convolutional blocks** (`self.model.features[-3:]`).
   - Reduced the learning rate to `1e-4` to prevent catastrophic forgetting while allowing the network to learn domain-specific industrial metal textures.
   - Result: Eliminated the oil spot vs. dent collision.

## ⚡ Hardware & Memory Optimization

Engineered to run on constrained hardware (6GB VRAM) without sacrificing batch size.

*   **Automatic Mixed Precision (AMP):** Integrated `torch.amp.autocast` and `GradScaler` in `optimized_train.py` and `pretrained_train.py` to cast operations to `float16`.
*   **Memory Footprint Reduction:** Implemented `filter(lambda p: p.requires_grad, model.parameters())` to prevent the Adam optimizer from tracking gradients on frozen layers.

## 📂 Repository Structure

```text
industrial_defect_engine/
├── api/
│   └── main.py                     # FastAPI zero-I/O inference endpoints
├── frontend/
│   └── app.py                      # Streamlit UI for drag-and-drop defect testing
├── notebooks/
│   └── 01_eda.ipynb                # Pixel intensity profiling & dimensionality analysis
├── src/
│   ├── config.py                   # Dynamic hyperparameter and path registry
│   ├── data_loader.py              # Custom WeightedRandomSampler pipeline (1-channel)
│   ├── pretrained_data_loader.py   # ImageNet-specific dataloader (224x224, 3-channel)
│   ├── optimized_train.py          # AMP-enabled training loop for Custom CNN
│   ├── pretrained_train.py         # Fine-tuning loop with dynamic optimizer filtering
│   └── models/
│       ├── base_cnn.py             # Custom 5-Layer CNN definition
│       └── pretrained_models.py    # ResNet18 & EfficientNet-B0 (Frozen/Unfrozen) architectures
└── artifacts/                      # Model weights (.pth) & training metrics

## 🚀 Execution & Deployment

The deployment layer separates the heavy PyTorch inference from the user interface, enabling modular scaling.

**1. Start the FastAPI Inference Server:**
```bash
uvicorn api.main:app --reload --port 8000
```

**2. Launch the Streamlit Client Interface:**
```bash
streamlit run frontend/app.py
```

## 🛠 Tech Stack

*   **Core DL:** PyTorch, Torchvision, Automatic Mixed Precision (AMP)
*   **Data Ops:** Scikit-Learn (Stratified Splits), NumPy, PIL
*   **Serving:** FastAPI, Uvicorn, Pydantic
*   **Frontend:** Streamlit