# Multimodal Emotion Recognition from Audio and Transcript

This project implements a deep learning pipeline for classifying human emotions using both audio data (as spectrograms) and text transcripts. It explores unimodal models (CNN for audio, RNN/Transformer for text) and multimodal fusion techniques.

## Objective

Design and train a deep learning model that classifies human emotions using:

1. **Audio data:** Processed visually as Mel spectrograms and fed into a Convolutional Neural Network (CNN).
2. **Text transcripts:** Generated via speech-to-text (OpenAI Whisper) or simulated based on emotion labels, then modeled using Recurrent Neural Networks (RNN) or Transformer-based models (DistilBERT).

The project aims to apply CNNs, RNNs, Transformers, and explore early and late fusion strategies for multimodal learning.

## Dataset

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):**
    - Kaggle Link: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
    - This project uses the audio-only speech portion.
    - 1440 speech clips from 24 actors (12 male, 12 female).
    - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.
    - **Note:** For the text modality, the option to use **simulated sentences** is implemented (`USE_SIMULATED_SENTENCES = True` by default) because the original RAVDESS dataset uses only two lexically-matched statements, providing limited textual cues for emotion.

## Project Structure

The project is structured into several phases:

1. **Data Loading and Preprocessing:**
        - Loads RAVDESS audio files.
        - Extracts emotion labels from filenames.
        - Generates text transcripts using OpenAI Whisper or uses predefined simulated sentences based on emotion labels.
        - Converts audio to Mel spectrograms with fixed dimensions.
        - Tokenizes text for NLP models.
        - Splits data into training, validation, and test sets.

2. **Unimodal Pipelines (Phase 1):**
        - **Audio CNN:** A 2D CNN is trained to classify emotions from spectrograms.
        - **Text RNN (LSTM):** An LSTM-based RNN is trained on text transcripts (simulated sentences by default).
        - **Text Transformer (DistilBERT - Bonus):** A DistilBERT model is fine-tuned on text transcripts.

3. **Multimodal Fusion (Phase 2):**
        - Features are extracted from the trained unimodal models.
        - **Early Fusion:** Audio and text features are concatenated and fed into a classification head. Separate early fusion models are trained for RNN-text and BERT-text features.
        - **Late Fusion:** Probabilities (or logits) from the unimodal audio and text models are averaged to make a final prediction. This is evaluated for both RNN-text and BERT-text.

## Files

- `your_script_name.ipynb` or `your_script_name.py`: The main Python script containing the entire implementation.
- `RAVDESS_PATH` (configurable): Path to the RAVDESS dataset directory (e.g., `./archive/audio_speech_actors_01-24/`).
- `AudioCNN_best.pth`: Saved weights for the best performing Audio CNN model.
- `TextRNN_best.pth`: Saved weights for the best performing Text RNN model.
- `TextBERT_best.pth`: Saved weights for the best performing Text BERT model.
- `EarlyFusion_RNN_best.pth`: Saved weights for the best early fusion model using RNN text features.
- `EarlyFusion_BERT_best.pth`: Saved weights for the best early fusion model using BERT text features.
- `label_encoder.pkl`: Saved Scikit-learn LabelEncoder for emotion labels.
- `ravdess_transcripts_simulated.pkl` (or similar): Cached transcripts if Whisper is used (name reflects if simulated sentences are primary).
- `emotion_recognition_report.docx`: A document summarizing the training and validation accuracies/metrics.

## Requirements

Key Python libraries needed:

- `torch`, `torchaudio`, `torchvision`
- `transformers` (Hugging Face)
- `librosa`
- `openai-whisper`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `soundfile`
- `Pillow`
- `python-docx` (for generating the report)

You can typically install these using pip:

```bash
pip install torch torchaudio torchvision transformers librosa openai-whisper numpy pandas scikit-learn matplotlib seaborn soundfile Pillow python-docx
```

You might also need ffmpeg for Whisper:

```bash
# On Linux
sudo apt update && sudo apt install ffmpeg
# On macOS
brew install ffmpeg
```

## How to Run

1. **Download Dataset:**
        - Download the RAVDESS Emotional Speech Audio dataset from Kaggle.
        - Extract it and place the `audio_speech_actors_01-24` folder inside an archive directory in the project root, or update `RAVDESS_PATH` in the script to point to your dataset location.

2. **Install Dependencies:** Install all required libraries as listed above.

3. **Configure Script (Optional):**
        - `USE_SIMULATED_SENTENCES`: Set to `True` (default) to use emotion-specific sentences for the text modality, or `False` to use Whisper-generated transcripts from the (limited) RAVDESS statements.
        - Adjust `EPOCHS_*` and `LEARNING_RATE_*` variables if needed.

4. **Execute the Script:**

        ```bash
        python your_script_name.py
        ```

        Or run the cells in `your_script_name.ipynb` if using a Jupyter Notebook.

        The script will:
        - Load and preprocess data.
        - Train and evaluate unimodal models.
        - Train and evaluate multimodal fusion models.
        - Train and evaluate the bonus Transformer text model and its fusion variants.
        - Save model weights, plots, and a results summary report.

## Expected Output

- Console logs showing training progress and evaluation metrics for each model.
- Saved model files (`.pth`).
- Plots for sample spectrograms and confusion matrices (displayed if running interactively, or can be saved).
- An `emotion_recognition_report.docx` file summarizing the key accuracy results.

## Potential Improvements & Future Work

- Experiment with more advanced CNN architectures (e.g., ResNet, EfficientNet adapted for spectrograms).
- Explore different Transformer models for text or even audio (e.g., Audio Spectrogram Transformer - AST).
- Implement more sophisticated fusion mechanisms (e.g., attention-based fusion, cross-modal attention).
- Perform extensive hyperparameter tuning for all models.
- Apply more advanced data augmentation techniques for both audio and text.
- Use a more diverse dataset for text-based emotion recognition if the goal is to capture lexical emotional cues.

## Before Running the Final Code

1. **Create Directory Structure:** Make sure you have the RAVDESS dataset in `./archive/audio_speech_actors_01-24/` relative to where you run the script, or update `RAVDESS_PATH`.
2. **Install All Dependencies:** Ensure every library mentioned is installed.
3. **Patience:** This script trains multiple complex models. It will take a significant amount of time to run, especially the BERT parts and if you increase epochs.
