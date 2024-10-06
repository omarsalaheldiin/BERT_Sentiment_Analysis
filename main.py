from model_handler import ModelHandler
from predictor import Predictor

def classify_text(model_handler):
    while True:
        user_input = input("Enter the text to classify (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        predicted_class_id = model_handler.predict(user_input)
        sentiment = "Positive" if predicted_class_id == 1 else "Negative"
        print(f"Predicted Class ID: {predicted_class_id}, Sentiment: {sentiment}")

def main():
    # Initialize model handler
    model_handler = ModelHandler()

    # Classify user input
    classify_text(model_handler)

if __name__ == "__main__":
    main()
