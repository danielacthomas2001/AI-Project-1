import streamlit as st
from transformers import pipeline


#title
st.title("Sentiment Text Analysis App")

#subtitle

st.markdown("## Sentiment Analyis App - Using 'streamlit'")
st.markdown("")
st.markdown("")
st.write("")
#brief description of app
st.write("Input a text and choose a pre-trained model to obtain its corresponding sentiment analysis. ")

default_text = "All those moments will be lost in time, like tears in rain."


with st.form(key='my_form'):
    text = st.text_input("Enter Your Text Here: ", value = default_text)
    model_name = st.selectbox('Select Model', ('bert-base-uncased', 'distilroberta-base', 'xlm-roberta-base', 't5-base'))
    submit_button = st.form_submit_button(label = 'Submit')

def sentiment_analysis(text, model_name):

    #Loads the selected model
    setiment_model = pipeline('sentiment-analysis', model = model_name )
    #performs the analysis
    result = setiment_model(text)[0]
    sentiment = result['label']
    score = result['score']
    if sentiment == 'LABEL_1':
        sentiment = 'negative'
    else:
        sentiment = 'positive'

    return sentiment, score

if submit_button:
    if text:
        sentiment, score = sentiment_analysis(text, model_name)
        st.write('Sentiment: ' + sentiment)
        st.write('Score: ' + str(score))
