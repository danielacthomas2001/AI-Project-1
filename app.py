import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def classify_comment(text, model_name):
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=1).detach().numpy()[0]

    # Get the highest toxicity class and its probability
    top_class_idx = probabilities.argmax()
    top_class_label = model.config.id2label[top_class_idx]
    top_class_prob = probabilities[top_class_idx]

    # Get the highest toxicity class among the 4 types and its probability
    toxic_types = ['obscene', 'threat', 'insult', 'identity_hate']
    toxic_types_idx = [model.config.label2id[label] for label in toxic_types]
    toxic_types_prob = probabilities[toxic_types_idx]
    top_toxic_type_idx = toxic_types_prob.argmax()
    top_toxic_type_label = toxic_types[top_toxic_type_idx]
    top_toxic_type_prob = toxic_types_prob[top_toxic_type_idx]

    return top_class_label, top_class_prob, top_toxic_type_label, top_toxic_type_prob

st.title("Toxic Comment Classifier")
st.markdown("## Input a text and choose a fine-tuned model to obtain its toxicity classification.")

default_text = "Your sample text goes here."

with st.form(key='my_form'):
    text = st.text_input("Enter Your Text Here: ", value=default_text)
    model_name = st.selectbox('Select Model', ('your-saved-model-directory',))
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if text:
        top_class_label, top_class_prob, top_toxic_type_label, top_toxic_type_prob = classify_comment(text, model_name)
        st.write('Toxicity class: ' + top_class_label)
        st.write('Probability: ' + str(top_class_prob))
        st.write('Highest toxicity class among the 4 types: ' + top_toxic_type_label)
        st.write('Probability: ' + str(top_toxic_type_prob))

        # Display the results in a table
        st.write(pd.DataFrame({
            'Tweet': [text],
            'Highest Toxicity Class': [top_class_label],
            'Probability': [top_class_prob],
            'Top Toxicity Class Among 4 Types': [top_toxic_type_label],
            'Probability for Top Toxicity Class Among 4 Types': [top_toxic_type_prob]
        }))
