🧠 Eye Disease Detection using CNN | DR & Cataract
A dual-pathology deep learning system for simultaneous detection of Diabetic Retinopathy (DR) and Cataracts from retinal fundus images, powered by VGG-16 and deployed on Hugging Face Spaces with a Streamlit frontend.

🔗 Live Demo on Hugging Face

📌 Project Highlights
🎯 Dual Detection: One model, two outputs—DR & Cataract predictions from a single retinal image.

🧠 Deep Learning Backbone: Pretrained VGG-16 architecture with customized dual-head classification.

⚙️ Multi-task Learning: Optimized loss functions enable simultaneous disease classification.

🧪 Datasets Used: APTOS 2019, MESSIDOR, and Kaggle-based Cataract slit-lamp dataset.

🖼️ Real-Time Interface: Hugging Face + Streamlit enables immediate upload & analysis (<1.2s).

📊 Performance Summary

Condition	Precision	Recall	F1-Score	Accuracy
Normal	0.95	0.95	0.95	0.97
DR (Perfect)	0.99	1.00	1.00	-
Cataract	0.95	0.94	0.95	-
Macro Avg.	0.97	0.97	0.97	0.97
✅ VGG-16 outperformed ResNet-50 and EfficientNet on all metrics.

🧬 Technologies Used
Backend: PyTorch, OpenCV, NumPy

Frontend: Streamlit (via Hugging Face Spaces)

Deployment: Hugging Face

Preprocessing: CLAHE, Gaussian Blur, Median Filtering

Model Structure:

Resized Input: 224x224 px

Dual output layers with Softmax for DR, Cataract, and Normal classification

📁 Project Structure
bash
Copy
Edit
📦 eye-disease-predictor
├── app.py             # Streamlit app interface
├── model.pth          # Trained VGG-16 weights
├── utils.py           # Preprocessing and helper functions
├── requirements.txt   # Environment dependencies
└── README.md          # You're reading it!
🧪 Datasets

Dataset	Description
APTOS 2019	3,662 fundus images with DR severity labels
MESSIDOR	1,200 retinal images (DR focused)
Custom Cataract	5,000 slit-lamp images, Cataract vs Normal
Sampling ensured class balance and excluded low-quality inputs.

📈 Training & Validation
Training/Validation/Test Split: 70/15/15

Convergence: Loss ~0.2 after 140 epochs

Framework: Google Colab + PyTorch

🧩 Unique Contributions
✅ Unified CNN Model: Reduces costs by 40%, suitable for low-resource clinics

🌐 Cloud-Hosted Tool: No setup required, directly usable via browser

⚖️ Adaptive Weighting: Prioritizes DR or Cataract classification depending on case confidence

🏥 Clinical Potential: Blueprint for PACS integration and real-world trials (planned Q2 2025)

🚧 Limitations & Future Work
Limitations
Sensitive to poor-quality fundus images (SNR < 30 dB)

Generalization across different populations requires federated learning

Cataract detection accuracy (~95%) slightly lags DR detection

Future Enhancements
🔍 Integrate Glaucoma detection (third pathology)

⚡ 8-bit quantization for edge deployment

🌍 Use federated learning for decentralized training

💡 Test Vision Transformers (ViT) and attention modules

📜 Citation & Report
This project was prepared as part of Advanced Database Topics (Winter 2025) under Dr. Shafaq Khan at Ontario Tech University.

Contributors:
Rushi Shah, Archi Brahmbhatt, Priya Vora, Divya Mistry, Arman Chowdhury, Firas Hussain Mohammed

📄 Full Report: DR_Cataract_9-Final_Report.docx

🌍 Impact
"Supporting UN SDG Goal 3: Health & Well-being – reducing vision screening costs by 60% and enabling early detection for 2.2 billion people globally."

