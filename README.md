# Pulmo: World-Class Lung Nodule Detection and Explainable Diagnosis Model 🫁🚀

Pulmo (LungXai) is a world-class artificial intelligence system developed for nodule detection, segmentation, and malignancy (cancer risk) analysis in lung CT scans. 

Unlike traditional "black-box" deep learning models, Pulmo utilizes a **Concept Bottleneck Model (CBM)** architecture. It bases its decisions on 8 clinical concepts used by expert radiologists (e.g., spiculation, margin, sphericity). This not only provides unmatched accuracy but also offers a level of **explainability** that sets a new standard in medical literature.

Combining 3D Vision Transformers (ViT) and U-Net architectures, the system also features super-fast 2.5D versions optimized through **Knowledge Distillation** techniques, making heavy 3D computations suitable for real-time clinical deployment.

---

## 📊 SOTA Performance & Concrete Metrics

Pulmo's architecture has been rigorously evaluated on the LUNA16 and LIDC-IDRI benchmarks, achieving scores that rival or exceed current State-of-the-Art (SOTA) literature. The concrete evaluation results and their derivations are extensively documented in our evaluation notebooks:

* **[Notebook 11] Comprehensive 3D Model Evaluation:**
    * **Detection AUC:** 0.998
    * **Malignancy AUC:** 0.986
    * **Segmentation Dice Score:** 0.857
    * *Details:* Includes patch-level metrics, calibration (Brier score), and scan-level FROC analysis based on LUNA16 official metrics.

* **[Notebook 13] 2.5D Distilled Student Model Evaluation:**
    * **Detection AUC:** 0.997
    * **Malignancy AUC:** 0.986
    * **Segmentation Dice Score:** 0.859
    * *Details:* Demonstrates that the lightweight 2.5D model maintains SOTA performance while significantly reducing VRAM usage and inference time.

These metrics demonstrate Pulmo's robust performance in both accuracy and interpretability compared to traditional baseline models in the domain.

---

## 📂 Step-by-Step Notebook Pipeline

The development pipeline of Pulmo is modularly designed across 15 notebooks, covering everything from data preparation to distillation:

### Data Preparation & Preprocessing
* **[Notebook 1] LIDC Malignancy & Segmentation Mask Extraction:** Spatially matches LUNA16 candidates with the LIDC-IDRI database to extract mean malignancy scores and 3D segmentation masks, creating the canonical dataset.
* **[Notebook 2] LUNA16 Multi-task Fine-tune Dataset Class:** Builds the dataset infrastructure for the multi-task model, configuring patch extraction, HU normalization, and negative sampling.
* **[Notebook 3] Patch Pre-computation & HDF5 Caching:** Overcomes training I/O bottlenecks by pre-computing patches offline and caching them into HDF5 files, boosting training speed by 50-100x.
* **[Notebook 4] LIDC Concept Extraction:** Lays the foundation for the Concept Bottleneck architecture by extracting radiological concepts (subtlety, calcification, spiculation, etc.) from the LIDC data.

### Model Architecture & Training
* **[Notebook 5] Concept Bottleneck Multi-task Model:** Constructs the architecture by combining a pretrained MAE-ViT-L encoder with 4 distinct heads (detection, concepts, malignancy, segmentation).
* **[Notebook 6] Diagnostic Head-Only Fine-tune:** Conducts a diagnostic, frozen-encoder training to analyze and resolve AUC plateaus.
* **[Notebook 7] Multi-task CBM Training:** End-to-end training of the hybrid architecture (frozen MAE-ViT-L and trainable 3D U-Net).
* **[Notebook 8] Training Loop & Evaluation:** Contains the full model training loop, metric tracking, checkpoint management, and Telegram log integration.
* **[Notebook 9] Advanced Training:** Maximizes performance by integrating Focal Loss, MixUp, and aggressive data augmentation techniques.

### Explainability & Evaluation
* **[Notebook 10] Explainability (Concept Bottleneck):** Analyzes how the model makes malignancy predictions through concept intervention, contribution analysis, and saliency maps.
* **[Notebook 11] Comprehensive Evaluation:** Computes patch-level AUC/Dice scores and scan-level standard LUNA16 FROC analysis on the test set.

### Optimization & Deployment
* **[Notebook 12] 2.5D Knowledge Distillation:** Distills knowledge from the heavy 3D teacher model to a lightweight 2.5D CNN student model for much faster execution in production environments.
* **[Notebook 13] Student (2.5D) Explainability:** Validates that the distilled 2.5D student model retains a pure Concept Bottleneck architecture and conducts its explainability analyses.

### Stage 1: Independent Nodule Detectors
* **[Notebook 14] Stage 1 Detector (3D) + Local Copy & H5 Precompute:** Performs 3D U-Net based nodule center detection to send candidates to Stage 2. Establishes local copy and H5 precompute infrastructure for speed and stability.
* **[Notebook 15] Stage 1 Detector v2 (Focal Loss & Hard Negative Mining):** An improved version of Notebook 14 that minimizes False Positives and maximizes performance using Focal Loss and aggressive hard negative mining.
