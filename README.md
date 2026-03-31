# retinal-scan-using-computer-vision
#  Diabetic Retinopathy Detection

**CNN-based 5-Class Severity Classification with Explainability**  
**Computer Vision (CSE3010) BYOP Project**  
**V. S. Ananya | 23BAI10927 | VIT Bhopal University**

## 📋 Project Overview

End-to-end **Computer Vision** pipeline that automatically detects and classifies Diabetic Retinopathy severity from retinal fundus images into **5 classes**: No DR, Mild, Moderate, Severe, and Proliferative.

Built using core CV techniques taught in CSE3010: image preprocessing, data augmentation, convolutional feature extraction, and post-hoc explainability (Grad-CAM + LIME).

Trained on the APTOS 2019 dataset. Demo achieves **75% training accuracy** (full dataset reaches ~75-80% validation accuracy).


##  Computer Vision Concepts Applied

| CV Concept                     | Implementation in Project                                      |
|--------------------------------|----------------------------------------------------------------|
| Image Preprocessing            | Resize (224×224), pixel normalization [0,1], RGB preservation |
| Data Augmentation              | Horizontal flip, rotation (±10°), zoom, brightness ±30%       |
| Convolutional Feature Extraction | 3 Conv2D blocks (32 -> 64 -> 128 filters)                     |
| Spatial Downsampling           | 2×2 MaxPooling2D (translation invariance)                      |
| Regularization                 | Dropout (0.25 after conv blocks, 0.50 after dense)            |
| Multi-Class Classification     | Softmax output + confidence scoring                            |
| Explainability                 | Grad-CAM (ResNet50 backbone) + LIME superpixel maps           |



##  System Pipeline

1. **Input** -> Retinal fundus image (JPEG)  
2. **Preprocessing** -> Resize + normalize (PIL + NumPy)  
3. **Augmentation** -> Geometric & photometric transforms  
4. **Feature Extraction** -> Custom 3-block CNN  
5. **Classification** -> Dense(128) -> Dense(5, softmax)  
6. **Explainability** -> Grad-CAM heatmap + LIME superpixels  
7. **Visualization** -> Prediction + confidence bar + attention maps

**Total parameters**: 12.9M


##  Results (Demo)

- **Training**: 75.0% accuracy after 5 epochs  
- **Validation**: 50.0% (small demo set)  
- Correctly classified: No DR and Proliferative DR  
- Explainability maps highlight clinically relevant regions (macula, vessels, lesions)

##  Explainability 

- **Grad-CAM**: Gradient-based heatmaps showing discriminative retinal regions  
- **LIME**: Black-box superpixel importance maps  

Both techniques directly demonstrate what the CNN "sees" -a key Computer Vision topic from the course.


##  Quick Start

```bash
git clone <your-repo>
cd dr-detection-cnn
pip install -r requirements.txt
python main.py --image samples/16_left.jpeg --explain
<img width="640" height="453" alt="image" src="https://github.com/user-attachments/assets/37074698-b03b-45e7-a4de-782d59da7043" />

<img width="1198" height="471" alt="image" src="https://github.com/user-attachments/assets/27142b52-50d2-4eeb-b74c-be7d891874ce" />
