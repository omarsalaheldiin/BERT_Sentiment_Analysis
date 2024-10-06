import streamlit as st
from model_handler import ModelHandler

# Initialize model handler
model_handler = ModelHandler()

def classify_text(text):
    predicted_class_id = model_handler.predict(text)
    sentiment = "Positive" if predicted_class_id == 1 else "Negative"
    return predicted_class_id, sentiment

def main():
    st.title("DEPI Graduation: Sentiment Analysis App")
    st.write("Enter a review to see if it's Positive or Negative.")

    user_input = st.text_area("Review Text", height=150)

    if st.button("Classify"):
        if user_input:
            predicted_class_id, sentiment = classify_text(user_input)
            st.success(f"Predicted Class ID: {predicted_class_id}")
            st.success(f"Sentiment: {sentiment}")
            #st.fail('message')
        else:
            st.error("Please enter some text.")

if __name__ == "__main__":
    main()
