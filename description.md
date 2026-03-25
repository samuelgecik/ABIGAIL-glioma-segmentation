## ABIGAIL — AI-Boosted Imaging for Glioma Analysis, Identification, and Localization

**TRL Level:** TRL 2–3 (from foundational research through experimental validation on real patient data)

**Partners:** Technical University of Košice (TUKE, lead) and Pavol Jozef Šafárik University in Košice (UPJŠ), Slovakia

---

### The Problem

Brain gliomas are the most common group of brain tumors and come in a wide spectrum of aggressiveness (WHO Grades 1–4). Glioblastoma (Grade 4) has a 5-year survival rate of just 7%. Current diagnostic workflows depend heavily on invasive procedures (biopsy or surgical resection) to confirm tumor subtype, and distinguishing true tumor progression from "pseudo progression" (a treatment side-effect mimicking tumor growth on MRI) remains a major clinical challenge. Tumor segmentation for radiotherapy planning is still semi-automatic and often requires manual corrections by radiation oncologists.

---

### Three Core Objectives

**SO1 — Brain Glioma Classification**
Develop a non-invasive classification model that distinguishes glioma subtypes (G2/G3/G4) from MRI scans combined with patient health records (PHR). The approach uses a **dual-stream multimodal neural network**:

- An **imaging stream** processes multiple MRI modalities (T1, T2, FLAIR) through deep CNNs (ResNet/DenseNet) or Vision Transformers with attention mechanisms.
- A **contextual data stream** processes structured patient data (age, sex, medical history) via fully connected networks and unstructured clinical text via NLP models (e.g., BERT).
- The two streams are fused using early and/or late fusion strategies to produce a unified classification.
- The model leverages **weakly supervised and semi-supervised learning** (including Multiple Instance Learning, self-training with pseudo-labels, and consistency regularization) to make use of the large volume of partially labeled medical data.

**SO2 — Treatment Response Assessment**
Build a model to assess glioma response to treatment, specifically differentiating between true progression and pseudo progression in Grade 3 and Grade 4 tumors, based on the RANO criteria. The approach uses a classification network (CNN/Vision Transformer) with an **optional registration module** (e.g., VoxelMorph) that spatially aligns pre-treatment and during-treatment MRI volumes before analysis. This aims to reduce the need for expensive PET-CT imaging.

**SO3 — Tumor and Organ-at-Risk Segmentation**
Introduce an automated segmentation model for delineating tumors and organs at risk of irradiation damage (optic nerves, brainstem, cochlea, hippocampus, etc.). The baseline architecture is **nnU-Net** (3D U-Net with task-specific pre/post-processing). The key innovation is **G-ABIGAIL**, a novel semi-supervised generative approach that:

1. Disentangles MRI scans into universal (healthy) features and tumor-specific features.
2. Reconstructs disease-free versions of scans and generates tumor-highlighting images.
3. Uses reverse domain translation to create synthetic diseased scans from healthy ones, augmenting the training data.

---

### Additional Methodological Innovations

- **Multitask Learning (MTL):** A single model is trained simultaneously on classification, segmentation, and grading tasks, allowing shared features to boost performance across all objectives.
- **Synthetic Data Generation for Underrepresented Grades:** Uses Conditional GANs (CGANs) or Conditional Stable Diffusion to generate realistic synthetic MRI images for rarer glioma types (e.g., G2), addressing class imbalance.
- **Explainability & Uncertainty:** Integration of LIME/SHAP for model interpretability and uncertainty estimation to provide clinicians with confidence measures alongside predictions.

---

### Data Sources

Primary data comes from the Louis Pasteur University Hospital (UNLP) in Košice through UPJŠ's tight research cooperation, with potential access to other Slovak hospitals and international biobanks like the UK Biobank.

---

### Expected Impact

- **Clinical:** Reduce reliance on invasive biopsy for glioma subtyping, eliminate unnecessary PET-CT scans, improve radiotherapy planning accuracy, and support personalized treatment decisions.
- **Societal:** Better patient outcomes, reduced procedural risks (hospital-acquired infections, surgical complications), and cost savings for healthcare systems.
- **Commercial:** Potential licensing to medical device manufacturers and healthcare IT firms; technology transfer supported by TUKE's Science Park TECHNICOM.
- **Scientific:** Open Access publications in top-ranked journals and conferences; contributions to medical imaging and deep learning methodology.

---

### Project Structure

The project is organized into **7 work packages**: WP1–2 (led by UPJŠ) focus on data collection and annotation; WP3–5 align with the three specific objectives (segmentation, classification, treatment assessment); dissemination and management activities are distributed across all WPs. The project follows Open Science practices, with results published on arXiv before formal peer review.

This is the project you previously worked on presentation materials for during your Caterpillar Digital interview preparation, Ivor — specifically the ABIGAIL brain MRI analysis project that was one of the four highlighted projects.