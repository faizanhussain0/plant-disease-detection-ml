# Plant Disease Detection using Deep Learning

## Author
Faizan Hussain  
GitHub: https://github.com/faizanhussain0  

---

## Project Overview
Plant diseases significantly impact agricultural productivity and farmers’ income.  
This project implements an image-based plant disease classification system using Deep Learning (CNN) to automatically identify plant diseases from leaf images.

The objective is to assist early disease detection and support better agricultural decision-making.

---

## Problem Statement
Manual identification of plant diseases is time-consuming and requires expert knowledge.  
An automated deep learning-based image classification system can help detect diseases quickly and accurately.

---

## Dataset
The model is trained on a publicly available plant leaf image dataset consisting of healthy and diseased plant leaves across multiple classes.

Due to large file size, the dataset is not included in this repository.  
You may use datasets such as the PlantVillage dataset or any similar plant disease dataset.

Example dataset structure:
dataset/
 ├── train/
 ├── validation/
 └── test/

---

## Technologies Used
- Python
- TensorFlow
- NumPy
- Scikit-learn
- OpenCV

---

## Approach
- Image preprocessing and normalization  
- Convolutional Neural Network (CNN) for feature extraction  
- Multi-class classification using Softmax activation  
- Model evaluation using validation data  

---

## Model Architecture
- Convolutional Neural Network (CNN)
- Convolution and pooling layers
- Fully connected layers
- Softmax output layer for multi-class prediction

---

## Training the Model
Run the following command to train the model:

python plant_disease_classification/trainer.py --train path_to_training_data --val path_to_validation_data --num_classes 38

---

## Model Output
After successful training, model checkpoints are saved in:

plant_disease_classification/ckpts/

---

## Inference / Prediction
The trained model can be used to classify new plant leaf images by providing the image path to the classifier module.

---

## Evaluation Metrics
- Accuracy
- Loss
- Confusion Matrix (optional enhancement)

---

## Key Learnings
- Image preprocessing for deep learning
- CNN-based image classification
- Multi-class classification handling
- Training and evaluating deep learning models
- Working with real-world image datasets

---

## Future Improvements
- Apply data augmentation
- Use transfer learning models such as ResNet or MobileNet
- Improve accuracy and generalization
- Deploy using Streamlit or Flask
- Enable real-time image prediction

---

## Disclaimer
This project is created for learning and academic purposes.  
The implementation is adapted from open-source references and extended with custom documentation and experimentation.


