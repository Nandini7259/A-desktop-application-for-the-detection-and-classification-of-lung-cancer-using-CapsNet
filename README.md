# 🫁 Desktop-Based Lung Cancer Detection and Classification using Capsule Network (CapsNet)

## 📌 Project Overview
This project is a desktop-based application developed for detecting and classifying lung cancer from CT scan images using a deep learning model called Capsule Network (CapsNet). The system provides a simple Streamlit-based interface where users can upload CT images and receive the predicted cancer stage based on tumor size.

This project was developed as an academic major project.

---

## 🎯 Objective
To design and implement a deep learning-based system that accurately classifies lung cancer stages from CT images and provides an easy-to-use desktop interface for users.

---

## ⚙️ Features
- CT image upload functionality
- Image preprocessing using OpenCV
- Lung cancer classification using Capsule Network (CapsNet)
- Stage prediction based on tumor size
- Interactive Streamlit user interface

---

## 🧠 Cancer Stage Classification Logic
The predicted tumor size determines the cancer stage as follows:

- Tumor size ≤ 30mm → Stage 1
- Tumor size ≤ 50mm → Stage 2
- Tumor size > 50mm → Stage 3

---

## 🛠 Technologies Used
- Python
- OpenCV
- TensorFlow
- Keras
- Streamlit

---

## 📂 Project Structure
Lung-Cancer-Detection-CapsNet/
│
├── app.py
├── requirements.txt
├── test/               ← Folder to store input CT images
├── images/             ← Folder to store README screenshots
├── README.md
└── other project files

---

## 🖼 Image Folder Instructions

For testing the model:
- Create a folder named **test**
- Store all CT scan input images inside the `test` folder
- The application will use images from this folder for prediction

For displaying screenshots in GitHub:
- Create a folder named **images**
- Store project screenshots (upload page, result page) inside the `images` folder

---

## 🚀 How to Run the Project

1. Clone the repository:
   git clone https://github.com/yourusername/Lung-Cancer-Detection-CapsNet.git

2. Navigate to the project directory:
   cd Lung-Cancer-Detection-CapsNet

3. Install the required libraries:
   pip install -r requirements.txt

4. Make sure your CT images are placed inside the `test` folder.

5. Run the Streamlit application:
   streamlit run app.py

(Replace app.py with your actual main file name if different.)

---

## 📷 Output Screenshots

### Upload Page
![Upload Page](images/upload.png)

### Prediction Result
![Prediction Result](images/result.png)

---

## ⚠️ Important Note
The trained model file (.h5) and dataset are not uploaded due to GitHub file size limitations (100MB restriction). The model can be shared separately upon request.

---

## 📌 Conclusion
This project demonstrates the application of Capsule Networks in medical image classification and provides a user-friendly desktop-based solution for lung cancer stage prediction.
