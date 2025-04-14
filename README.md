# NeuroDetect

NeuroDetect is an AI-powered brain cancer detection system that leverages deep learning techniques to classify MRI scans. This project employs Convolutional Neural Networks (CNNs) to differentiate between healthy and cancerous brain tissues, aiming to assist medical professionals in early and accurate diagnosis.

Project Overview
- Objective: To develop a robust and accurate deep learning model for detecting brain cancer from MRI images.
-Technologies Used: Python, TensorFlow, Keras, VGG16, OpenCV, Matplotlib.
-Dataset: The project uses MRI scans split into training and testing datasets to train and evaluate the model.
Features
- Data Augmentation: Utilizes extensive data augmentation techniques to improve model generalization, including rotation, width/height shift, shear, zoom, and flip.
- Transfer Learning: Implements the VGG16 model pre-trained on ImageNet as the base, fine-tuning it for brain cancer detection.
- Model Regularization: Applies dropout and L2 regularization to prevent overfitting and improve model robustness.
- Performance Optimization: Includes callbacks such as early stopping, learning rate reduction, and model checkpointing to optimize training.
Results
Achieved significant accuracy improvements over the training epochs, with the final model exhibiting strong learning on training data and reasonable generalization on validation data.
Training accuracy reached close to 95%, while validation accuracy stabilized around 75%, indicating good, but improvable, performance on unseen data.

Data - kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

To run project:
- Place your MRI scan images in the dataset/Training and dataset/Testing directories.
- Run the FinalModel.py file to create the model.
- Run the main.py to see the model results on the test data.

Future Work:
- Enhance the dataset with more MRI scans and additional augmentation techniques.
- Experiment with other CNN architectures such as ResNet, DenseNet, and Inception.
- Implement more sophisticated preprocessing steps and segmentation techniques.
- Explore ensemble methods to improve model accuracy and robustness further.
