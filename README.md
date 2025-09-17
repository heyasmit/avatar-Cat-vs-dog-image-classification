🐶🐱 Cat vs Dog Image Classifier using CNN
📌 Project Overview
This project builds a Convolutional Neural Network (CNN) from scratch to classify images of cats and dogs. Using the Kaggle Dogs vs. Cats (filtered 25K) dataset, the model is trained on labeled image data and includes a Gradio interface for live predictions using uploaded images or webcam input.

⚙ Technologies Used
Python
TensorFlow / Keras
OpenCV
Matplotlib
Gradio
Kaggle API
Google Colab
📁 Dataset
The dataset used is:
Dogs vs. Cats - filtered 25K on Kaggle

Total: ~25,000 images
Balanced classes of cat and dog images
Split into /train and /test folders
🚀 How It Works
Download Dataset:
Uses Kaggle API to download and unzip the dataset

Data Loading:
Images are loaded using image_dataset_from_directory with image resizing to 256x256

Preprocessing:

Normalized pixel values to [0, 1]
Batched and shuffled datasets
CNN Model Architecture:

Conv2D → BatchNorm → MaxPooling (×3)
Flatten → Dense(128) → Dropout
Dense(64) → Dropout → Output layer (sigmoid)
Training:

Optimizer: Adam
Loss: Binary Crossentropy
Accuracy metric tracked over 10–25 epochs
Evaluation:

Accuracy and loss plotted using Matplotlib
Model performance visualized over epochs
Prediction on Custom Images:

OpenCV is used to read and resize custom images
Output is a confidence score with a class prediction
Gradio Interface (Optional):

Upload or webcam input
Displays prediction and confidence score
📊 Results
Achieved ~85–90% accuracy with proper training
Consistent results on both training and validation sets
Can detect dogs and cats from custom images
🖼 Example Output
Prediction:
Dog (Prediction Score: 0.9586)
Cat (Prediction Score: 0.0201)

✅ Requirements
Install required libraries using:

pip install tensorflow opencv-python matplotlib gradio numpy
📂 How to Run
Upload your kaggle.json to authenticate with Kaggle
Run the notebook step by step in Google Colab
Use the last section to test on custom images or launch the Gradio app
🧑‍💻 Developed By
ASMIT SRIVASTAVA
Student at United College of Engineering and Research, Prayagraj
