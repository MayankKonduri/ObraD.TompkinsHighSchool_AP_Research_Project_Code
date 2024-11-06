# AP_Research_Project_Code (August 2023 - May 2024)

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

## Evaluation
To evaluate the model's performance, we use scikit-learn's classification report metrics, which can be accessed through the `evaluate` method. The metrics provide insights into the model's precision, recall, and F1 score, allowing a detailed assessment of its accuracy and robustness.

## Requirements
- Python 3.x
- Tweepy
- Keras
- TensorFlow
- NumPy
- Pandas
- scikit-learn

## Installation
1. Clone the repository:

## Running the Code
```bash
git clone https://github.com/yourusername/ObraD.Tompkins_AP_Research_Project_Code
ls
python3 [Code To Be Compiled]
