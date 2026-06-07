# Group13 Happywhale Final Project

<img width="948" height="784" alt="image" src="https://github.com/user-attachments/assets/adb7d5e5-d961-4dcd-b11f-d0a63ffcfb48" />

## Introduction

Tracking individual whales and dolphins is essential for ecological and conservation research, but traditional methods such as physical tagging and manual image identification are often expensive, time-consuming, and difficult to scale. Recent advances in computer vision make it possible to automatically identify individual animals from photographs, similar to human facial recognition systems.

This project explores deep learning approaches for whale and dolphin identification using image-based recognition. By focusing on distinguishing individual animals within a small set of cetacean species, the goal is to develop an accurate and scalable solution that can support wildlife monitoring, behavioral studies, and conservation efforts.

## Related Work

This project builds upon prior research in species classification, cetacean identification, and deep metric learning. Several key works influenced the design of our approach.

### Species Classification

One of the earliest large-scale applications of deep learning for wildlife monitoring was the work of Norouzzadeh et al. (2018), which used convolutional neural networks (CNNs) to automatically identify and classify animals captured by camera traps in the Serengeti. Although their work focused on species classification rather than individual identification, it demonstrated that deep CNNs can effectively learn discriminative visual features from large and noisy wildlife datasets.

### Cetacean Identification

Trotter et al. proposed a dedicated pipeline for automated cetacean photo-identification. Their approach combined dorsal fin segmentation, image preprocessing, and a Siamese Neural Network (SNN) to compare individuals within a learned embedding space. This work provided important insights into formulating cetacean identification as a similarity-learning problem rather than a traditional classification task.

### EfficientNet and ArcFace

Our implementation is primarily inspired by the winning solution of the Kaggle HappyWhale competition. The approach utilizes EfficientNet as a feature extractor and ArcFace as the training objective for learning discriminative embeddings.

* **EfficientNet** introduces compound scaling of network depth, width, and image resolution, achieving strong accuracy while maintaining computational efficiency.
* **ArcFace** improves feature embedding quality by introducing an angular margin loss, encouraging samples from the same individual to cluster closely while increasing separation between different individuals.

Together, these techniques have demonstrated state-of-the-art performance in both facial recognition and fine-grained image identification tasks, making them well-suited for whale and dolphin individual recognition.

## Methods

### Project Overview

Due to the limited project timeline, we adopted the winning Kaggle HappyWhale solution, **Preferred Dolphin (PD)**, as our baseline. This allowed us to focus on reproducing a strong-performing approach, conducting targeted experiments, and optimizing training configurations rather than exploring numerous architectures from scratch.

The project was carried out in four stages:

1. Reproducing the PD baseline and reviewing relevant literature.
2. Running experiments and fine-tuning model configurations.
3. Training final production models and preparing project materials.
4. Performing additional optimization and ensemble evaluation.

### Data Processing Pipeline

The original competition dataset was combined with several auxiliary datasets containing dorsal fin and full-body annotations. These datasets were used to train a YOLO-based detector for automated segmentation and bounding box extraction.

The resulting cropped images were then used for training the identification model, allowing the network to focus on the most informative visual regions while reducing background noise.

### Model Architecture

<img width="512" height="288" alt="image" src="https://github.com/user-attachments/assets/1c9d141a-2263-45de-980e-cf0e41de0419" />


The final architecture follows the general design of the Preferred Dolphin solution:

```text
Input Image
      ↓
EfficientNet Backbone (B6/B7)
      ↓
GeM Pooling
      ↓
Batch Normalization
      ↓
ArcFace Classification Head
      ↓
Cosine Similarity Loss
```

Key components include:

* **EfficientNet B6/B7** as the feature extraction backbone.
* **GeM Pooling** for learnable feature aggregation.
* **Batch Normalization** to stabilize training.
* **ArcFace Loss** to learn discriminative embeddings for individual identification.

### Training Strategy

Multiple EfficientNet configurations were evaluated using image resolutions ranging from **640×640** to **1440×1440**. Most final models were trained for **30 epochs**, balancing performance and computational cost.

During inference, predictions from several high-performing models were combined using an **ensemble approach**, improving robustness and overall identification accuracy compared to any single model.


## Results

All experiments were conducted using Google Colab G4 GPUs, with approximately **300 GPU-hours** of total training and evaluation. Performance was measured using both validation accuracy and Kaggle submission scores.

### Backbone Comparison

Experiments focused primarily on **EfficientNet-B6** and **EfficientNet-B7** architectures across multiple image resolutions.

The strongest single model was the **EfficientNet-B6** configuration trained at **1024×1024** and fine-tuned at **1280×1280**, achieving an individual validation accuracy of **81.28%**.

### Effect of Resolution

Input resolution was found to be one of the most important factors affecting performance:

* Lower resolutions removed fine-grained visual features needed for individual identification.
* Higher resolutions generally improved performance when sufficient training time was provided.
* Increasing resolution without adequate retraining often led to limited gains or performance degradation.
* The best-performing models utilized resolutions between **1024×1024** and **1280×1280**.

### Ensemble Performance

Several ensemble configurations were evaluated. Combining strong and weak models generally reduced performance, while combining models with similar standalone performance produced better results.

The best ensemble consisted of:

* EfficientNet-B6 (1024×1024 → 1280×1280 fine-tuned)
* EfficientNet-B7 (1280×1280)

This ensemble leveraged differences in architecture and training history to learn complementary feature representations and produced the strongest overall Kaggle submission.

### Key Findings

* Input resolution significantly impacts individual identification performance.
* Bounding box quality and preprocessing are critical for extracting discriminative features.
* Larger models do not automatically outperform smaller models without sufficient training and tuning.
* Ensembling is most effective when combining multiple strong and diverse models.
* The final EfficientNet-B6/B7 ensemble achieved a **Gold-level result** in the HappyWhale Kaggle competition.

### Full Competition Submission Results

### Kaggle Submission Results

| Model Configuration                                                                                        | Public Score | Private Score | Ranking        |
| ---------------------------------------------------------------------------------------------------------- | ------------ | ------------- | -------------- |
| EfficientNet B7 (768×768, 30 epochs)                                                                       | 0.69942      | 0.66420       | 874            |
| EfficientNet B6 (1024×1024, 30 epochs) + EfficientNet B7 (768×768, 30 epochs)                              | 0.75921      | 0.73166       | 491            |
| EfficientNet B7 (1024×1024, 30 epochs + 1024x1024, 15 epochs fine-tuning)                                  | 0.81531      | 0.78300       | 108 (Bronze)   |
| EfficientNet B6 (1024×1024, 30 epochs + 824×824, 15 epochs)                                                | 0.82330      | 0.79438       | 81 (Bronze)    |
| EfficientNet B7 (1024×1024, 30 epochs, fine-tuned)                                                         | 0.82330      | 0.79639       | 77 (Silver)    |
| EfficientNet B7 (1024×1024, 30 epochs + 1280×1280, 10 epochs)                                              | 0.82787      | 0.79912       | 72 (Silver)    |
| EfficientNet B6 (1024×1024, 30 epochs + 1440×1440, 15 epochs)                                              | 0.83271      | 0.80222       | 66 (Silver)    |
| EfficientNet B6 (1024×1024, 30 epochs) + EfficientNet B7 (1024×1024, 30 epochs)                            | 0.82849      | 0.80624       | 62 (Silver)    |
| EfficientNet B6 (1024×1024, 30 epochs + 1024x1024, 15 epochs fine-tuning)                                  | 0.83467      | 0.81379       | 53 (Silver)    |
| EfficientNet B6 (1024×1024, 30 epochs)                                                                     | 0.83678      | 0.81584       | 49 (Silver)    |
| EfficientNet B6 (1024×1024, 30 epochs + 1280×1280, 15 epochs + 1356×1356, 10 epochs)                       | 0.84987      | 0.82945       | 26 (Silver)    |
| EfficientNet B7 (1280×1280, 44 epochs, fine-tuned)                                                         | 0.85034      | 0.82979       | 26 (Silver)    |
| EfficientNet B6 (1024×1024, 30 epochs + 1280×1280, 15 epochs)                                              | 0.85019      | 0.82991       | 26 (Silver)    |
| **EfficientNet B6 (1024×1024, 30 epochs + 1280×1280, 15 epochs) + EfficientNet B7 (1280×1280, 30 epochs)** | **0.88145**  | **0.85661**   |  **10 Gold**   |

### Best Result

The strongest submission was an ensemble consisting of:

* EfficientNet B6 (1024×1024, 30 epochs + 1280×1280, 15 epochs)
* EfficientNet B7 (1280×1280, 44 epochs, fine-tuned)

This ensemble achieved:

| Metric                    | Value          |
| ------------------------- | -------------- |
| Public Leaderboard Score  | **0.88145**    |
| Private Leaderboard Score | **0.85661**    |
| Final Ranking             | **10th Place** |
| Medal                     |    **Gold**    |

The results demonstrate that carefully selected ensembles consistently outperformed individual models. The highest-performing solution combined two independently strong backbones trained at high resolutions, allowing the ensemble to leverage complementary feature representations and achieve the best overall competition performance.

### Submission Snapshot

<img width="1310" height="142" alt="image" src="https://github.com/user-attachments/assets/8fe7ec84-14f6-4bc7-9428-81d70cd9b0c2" />


## Conclusion

### Key Takeaways

This project highlights that **input resolution, preprocessing quality, and overfitting control** are the most critical factors influencing performance in cetacean individual identification.

A major limiting factor throughout the project was **compute constraints**, which restricted extensive experimentation with higher resolutions and longer training schedules. As a result, a significant portion of training time was dedicated to reproducing baseline results and optimizing within limited resources.

### Importance of Preprocessing

Dataset preprocessing—particularly the YOLO-based segmentation pipeline—had a substantial impact on overall performance. This suggests that **early-stage errors in segmentation propagate and are amplified downstream**, limiting the effectiveness of architectural improvements.

In contrast, modifications to the model architecture itself produced relatively small performance changes, indicating that the chosen EfficientNet + ArcFace pipeline was already well-optimized for this task.

### Role of Ensembling

Ensembling proved to be one of the most important contributors to final performance. In particular, combining models trained at **different input resolutions** led to significant gains. These models learned complementary feature representations, allowing the ensemble to compensate for individual weaknesses and improve overall robustness.

### Future Work

Several promising directions for further improvement were identified:

* **Improved filtering of training data**, using higher-confidence segmentation outputs to reduce noise.
* **Additional task-specific heads**, such as tail fluke detection, facial regions, or texture-based features for species with distinctive markings.
* **Alternative loss formulations**, including:

  * Orthogonal dictionary learning to better separate species in feature space.
  * Class-conditioned anisotropic feature scaling to improve inter-class separation in embedding space.
* **Refinement of embedding learning**, particularly to improve fine-grained individual discrimination while maintaining strong species-level classification.

### Final Remarks

Overall, we consider the project to be successful, achieving a **Gold medal-level performance** in the HappyWhale Kaggle competition. The results demonstrate that careful preprocessing, resolution scheduling, and ensembling strategies are more impactful than large architectural changes in this problem setting.
