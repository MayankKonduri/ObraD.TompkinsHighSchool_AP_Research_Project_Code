# AP_Research_Project_Code (August 2023 - May 2024) 

> **Important Note:** I received a 5/5 on the AP Research Exam (*The Highest Possible Score*) after submitting my paper and presenting via an oral defense.

**Update:** Paper has been published in **IJSRET Journal** in **December 2024**.
![IJSRET Certificate of Publication](Users/mayank/Downloads/IJSRET_Paper_Information/IJSRET_CertificateOfPublication.png)

## AI Tweet Classifier

## Overview
This repository contains the code for my **AP Research Project**, where I developed an **AI Tweet Classifier**. This project leverages **Tweepy** to fetch tweets and a **neural network** to classify whether a tweet was generated by AI or a human. The goal of this research was to explore the distinguishing characteristics of AI-generated text on social media and build a model capable of identifying these patterns in real-time tweets.

## Project Goals
The primary objective of this AP Research project is to:
- Investigate patterns in AI-generated text versus human-authored content on Twitter.
- Develop a reliable and efficient model to classify tweets based on these patterns.
- Contribute to ongoing research on AI detection, misinformation, and authenticity in online communications.

## Model Architecture
The classifier is designed with a **neural network** architecture suitable for processing text data and extracting features relevant to distinguishing AI-generated content. The main components include:

- **Embedding Layer:** Transforms the input text into dense vectors to capture semantic meaning.
- **LSTM Layers:** Capture sequential dependencies in the text data, making the model effective at identifying subtle differences in text generation style.
- **Dense Layer:** Outputs a probability score indicating the likelihood of a tweet being AI-generated.


<img width="920" alt="Screenshot 2024-11-05 at 9 02 51 PM" src="https://github.com/user-attachments/assets/117763c8-dc82-45af-95b8-d7ca7a5b3ce2">


## Evaluation
To evaluate the model's performance, we use scikit-learn's classification report metrics, which can be accessed through the `evaluate` method. The metrics provide insights into the model's precision, recall, and F1 score, allowing a detailed assessment of its accuracy and robustness.


<img width="930" alt="Screenshot 2024-11-05 at 9 03 10 PM" src="https://github.com/user-attachments/assets/aed180fd-e3cd-408d-9fc3-903ab790c20e">


## Requirements
- Python 3.x
- Tweepy
- Keras
- TensorFlow
- NumPy
- Pandas
- scikit-learn

## Running the Code
```bash
git clone https://github.com/yourusername/ObraD.Tompkins_AP_Research_Project_Code
ls
python3 [Code To Be Compiled]
