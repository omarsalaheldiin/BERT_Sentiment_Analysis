# Sentiment Analysis Project

This project is a sentiment analysis application that uses the DistilBERT model to classify text reviews as Positive or Negative. It includes both a web interface using Streamlit and a command-line interface.

## Features
- **Web Interface**: Enter text and get instant sentiment classification.
- **Command-Line Interface**: Classify text through the terminal.
- **Accurate Predictions**: Uses the DistilBERT model fine-tuned on the SST-2 dataset.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install the required packages**:
   Make sure you have Python 3.6 or higher. Run:
   ```bash
   pip install streamlit torch transformers scikit-learn
   ```

## Usage

### Web Interface
Run the Streamlit app:
```bash
streamlit run app.py
```
Open the URL in your browser, enter review text, and see the sentiment classification.

### Command-Line Interface
Run the command-line interface:
```bash
python main.py
```
Input the text you want to classify, and type 'exit' to quit.

## Project Structure
```
sentiment-analysis/
├── app.py              # Streamlit web app
├── main.py             # Command-line interface
├── model_handler.py    # Model loading and prediction
└── predictor.py        # Model evaluation
```

## Model Description
The project uses the [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model, optimized for sentiment classification.

