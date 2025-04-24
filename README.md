# 🧠 Eye Disease Detection using CNN | DR & Cataract

A dual-pathology deep learning system for **simultaneous detection of Diabetic Retinopathy (DR)** and **Cataracts** from retinal fundus images, powered by **VGG-16** and deployed on **Hugging Face Spaces** with a **Streamlit** frontend.

🔗 **[Live Demo on Hugging Face](https://huggingface.co/spaces/RushiShah19/eye-disease)**

---

## 📌 Project Highlights

- 🎯 **Dual Detection:** One model, two outputs—DR & Cataract predictions from a single retinal image.
- 🧠 **Deep Learning Backbone:** Pretrained VGG-16 architecture with customized dual-head classification.
- ⚙️ **Multi-task Learning:** Optimized loss functions enable simultaneous disease classification.
- 🧪 **Datasets Used:** APTOS 2019, MESSIDOR, and Kaggle-based Cataract slit-lamp dataset.
- 🖼️ **Real-Time Interface:** Hugging Face + Streamlit enables immediate upload & analysis (<1.2s).

---

## 📊 Performance Summary

| Condition        | Precision | Recall | F1-Score | Accuracy |
|------------------|-----------|--------|----------|----------|
| **Normal**       | 0.95      | 0.95   | 0.95     | 0.97     |
| **DR (Perfect)** | 0.99      | 1.00   | 1.00     | -        |
| **Cataract**     | 0.95      | 0.94   | 0.95     | -        |
| **Macro Avg.**   | 0.97      | 0.97   | 0.97     | **0.97** |

✅ **VGG-16** outperformed ResNet-50 and EfficientNet on all metrics.

---

## 🧬 Technologies Used

- **Backend**: PyTorch, OpenCV, NumPy  
- **Frontend**: Streamlit (via Hugging Face Spaces)  
- **Deployment**: Hugging Face  
- **Preprocessing**: CLAHE, Gaussian Blur, Median Filtering  
- **Model Input**: 224 × 224 pixels  
- **Output**: Multi-label classification for DR, Cataract, Normal

---

## 🧪 Datasets

| Dataset           | Description                                     |
|-------------------|-------------------------------------------------|
| **APTOS 2019**     | 3,662 fundus images with DR severity labels     |
| **MESSIDOR**       | 1,200 retinal images (DR focused)               |
| **Cataract Set**   | 5,000 slit-lamp images (Cataract vs Normal)     |

✔️ Preprocessed for class balance and image clarity.

---

## 📈 Training & Validation

- **Split**: 70% Train / 15% Val / 15% Test  
- **Loss**: Converges to ~0.2 around epoch 140  
- **Framework**: Google Colab + PyTorch  
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 🧩 Unique Contributions

- ✅ Unified detection model reduces cost and complexity
- 🌍 Deployable in low-resource healthcare settings
- 🧠 Multi-task CNN enables shared learning for related diseases
- 💻 Real-time diagnosis via browser-based interface
- 🌐 Fully open-source for community collaboration

---

## 🚧 Limitations & Future Work

### Limitations
- Sensitive to poor-quality images, especially in low-light or blurred scans
- May need retraining for generalizability across hospitals/devices
- VGG-16 backbone is relatively heavy for mobile/edge deployment

### Future Enhancements
- Implement **attention modules** to boost cataract detection
- Explore **Vision Transformers (ViT)** for robust feature learning
- Add **Glaucoma** as a third pathology
- Optimize via **model quantization** for edge deployment
- Apply **federated learning** across global clinics for broader generalizability

---




