#  Keyword Spotting System

This project implements a complete keyword spotting system that detects spoken words like "yes", "no", "stop", "go", etc., using MFCC features and a convolutional neural network (CNN).

---

##  Features

- Extracts MFCC and Mel features from speech samples using various window sizes and overlaps
- Trains a CNN to classify audio into 10 keywords
- Evaluates model accuracy across all window/overlap combinations
- Deploys a real-time, mic-based inference system with voice activity detection (VAD)

---

## 📁 Project Structure

| File / Folder                             | Description                                  |
|------------------------------------------|----------------------------------------------|
| `feature_extraction_pipeline.ipynb`      | Extract MFCC and Mel features from audio     |
| `Model_training and perfomance.ipynb`    | Train CNN, evaluate accuracy for all configs |
| `results_summary.csv`                    | Stores accuracy results for all experiments  |
| `live_inference.ipynb`                   | Real-time keyword detection via microphone   |
| `best_model.pth`                         | Trained PyTorch model (MFCC-based)           |
| `requirements.txt`                       | Python dependencies                          |

---

##  Supported Keywords

The model detects the following 10 keywords:

```
yes, no, up, down, left, right, on, off, stop, go
```

---

##  How to Run Live Inference

1.  Install requirements:

```bash
pip install -r requirements.txt
```

2.  Run the live system:

   - Option 1: Open `live_inference.ipynb` in Jupyter Notebook
   - Option 2: Convert to `.py` and run in terminal

3.  Speak one of the supported keywords clearly — you’ll see predictions printed in the terminal.

---

##  Dataset

This system was developed using the [Kaggle TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) dataset, which includes labeled 1-second audio samples.

---

##  Report Summary

- Experimented with 10–100ms window sizes and 0–75% overlaps
- Compared MFCC vs. Mel performance across all configurations
- Evaluated accuracy and summarized results in `results_summary.csv`
- Final system deployed with live mic input and real-time predictions

---