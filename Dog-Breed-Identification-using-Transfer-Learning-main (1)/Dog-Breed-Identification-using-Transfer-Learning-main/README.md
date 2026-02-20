# Dog Breed Identification Using Transfer Learning

## 📘 Overview
The **Dog Breed Identification** project leverages **Transfer Learning** to classify dog breeds from images. By using pre-trained **Convolutional Neural Networks (CNNs)**, the system can accurately recognize multiple dog breeds, even with a limited dataset.

This project demonstrates a deep learning-based classification system that can be applied to:

- Pet care and adoption platforms  
- Veterinary diagnostics  
- Animal welfare and breed research  
- Educational tools for dog breed recognition  

---

## 🧠 Key Features
- 🚀 High-accuracy classification of dog breeds  
- 🧩 Transfer learning with pre-trained CNN models (e.g., VGG19, Xception)  
- 🌐 Web-based interface built using Flask for easy image uploads and live predictions  
- 🧠 Data augmentation for better generalization on limited datasets  
- 📱 TensorFlow Lite export for deployment on mobile devices or edge applications  

---

## ⚙️ Technical Stack
- **Languages:** Python  
- **Libraries & Frameworks:** TensorFlow, Keras, OpenCV, Flask, NumPy  
- **Tools:** Google Colab, Kaggle, Roboflow (optional)  
- **Model Type:** Transfer Learning (CNN-based)  
- **Deployment:** Flask Web App, TensorFlow Lite  

---

## 🧩 Project Architecture

### 1. User Interaction & Data Input
- Users upload dog images through a **Flask web interface**.

### 2. Data Collection & Preparation
- Dataset of multiple dog breeds, organized into **training** and **testing** folders.

### 3. Data Preprocessing
- Image resizing, normalization, and **data augmentation** including:  
  - Rotation  
  - Flip  
  - Shift  
  - Zoom  

### 4. Model Building
- Fine-tuning **pre-trained CNN models** (e.g., VGG19) for **multi-class classification** of dog breeds.  

### 5. Evaluation
- Metrics include:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

### 6. Deployment
- Integration with Flask for **real-time breed prediction**.  

### 7. Visualization
- Display **predicted breed**, **confidence score**, and **uploaded image** on the Flask UI.  

---

## 📂 Folder Structure 
Dog-Breed-Identification/
│
├── app.py # Flask backend
├── dogbreed.h5 # Trained CNN model
├── requirements.txt # Python dependencies
├── templates/
│ ├── index.html
│ ├── predict.html
│ └── output.html
├── static/
│ ├── css/
│ └── images/
├── dataset/
│ ├── train/
│ └── test/
└── README.md


## 🚀 How to Run
1. Clone the repository:  
git clone <repository-url>

Install dependencies:
pip install -r requirements.txt

Run the Flask app:
python app.py

Open a web browser and go to:
http://127.0.0.1:5000/

## 🏆 Results
- The model achieved **high accuracy** on the test dataset, effectively distinguishing between multiple dog breeds.  
- **Confusion matrices** and **classification reports** confirm reliable predictions across common dog breeds.  
- Real-time predictions in the **Flask web application** are fast and user-friendly, making the system practical for end-users.  

---

## 🔮 Future Scope
- **Expand the dataset** to include rare dog breeds for broader recognition.  
- **Integrate object detection** to handle images containing multiple dogs.  
- **Deploy as a mobile app** using TensorFlow Lite for on-device predictions.  
- **Use ensemble models** to further improve prediction accuracy.  
- **Integrate with adoption platforms or veterinary tools** for real-world applications.  

## 👥 Team Details

**Team ID:** LTVIP2025TMIDS63802

**Team Leader:** Pammi Likhitha  

**Team Members:**  
- Puppalla Rama Krishna  
- Shaik Shamila  
- Shaik Mahaboob Subhani  

## 📌 References
1. Chollet, F. (2018). *Deep Learning with Python* (2nd Edition). Manning Publications.  
2. Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556.  
3. TensorFlow Documentation: [Transfer Learning & ImageDataGenerator](https://www.tensorflow.org/tutorials/images/transfer_learning)  
4. Flask Official Documentation: [https://flask.palletsprojects.com/en/2.3.x/](https://flask.palletsprojects.com/en/2.3.x/)

## 🎥 Demonstration Video

You can view the project demonstration video using the following link:  

[Watch Demo](https://drive.google.com/drive/folders/1sAp7n_9iDTkqsfwue6ozlxReEmC1urOT?usp=drive_link)

