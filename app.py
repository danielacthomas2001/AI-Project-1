import streamlit as st
from transformers import pipeline

# Set the title of the web application
st.title("Sentiment Text Analysis App")

# Set the subtitle of the web application
st.markdown("## Sentiment Analyis App - Using 'streamlit'")

# Add some empty lines for formatting
st.markdown("")
st.markdown("")
st.write("")

# Describe the purpose of the web application
st.write("Input a text and choose a pre-trained model to obtain its corresponding sentiment analysis.")

# Set a default text for the input field
default_text = "All those moments will be lost in time, like tears in rain."

# Create a form to input text and select a model
with st.form(key='my_form'):
    text = st.text_input("Enter Your Text Here: ", value=default_text)
    model_name = st.selectbox('Select Model', ('bert-base-uncased', 'distilroberta-base', 'xlm-roberta-base', 't5-base'))
    submit_button = st.form_submit_button(label='Submit')

def sentiment_analysis(text, model_name):
    # Loads the selected model
    setiment_model = pipeline('sentiment-analysis', model=model_name)
    
    # Performs the sentiment analysis on the input text
    result = setiment_model(text)[0]
    sentiment = result['label']
    score = result['score']

    return sentiment, score

# Display the sentiment analysis result when the submit button is clicked
if submit_button:
    if text:
        sentiment, score = sentiment_analysis(text, model_name)
        st.write('Sentiment: ' + sentiment)
        st.write('Score: ' + score)
