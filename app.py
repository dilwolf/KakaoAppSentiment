import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_data
def get_model():
    tokenizer = BertTokenizer.from_pretrained('Dilwolf/Kakao_app-kr_sentiment')
    model = BertForSequenceClassification.from_pretrained("Dilwolf/Kakao_app-kr_sentiment")
    return tokenizer, model

tokenizer, model = get_model()

# Define the "How to Use" message
how_to_use = """
**How to Use**
1. Enter text in the text area
2. Click the 'Analyze' button to get the predicted sentiment of the input text
"""

# Functions
def main():
    st.title("Kakao App Review Sentiment Analysis using BERT")
    st.subheader("Dilshod's Portfolio Project")

    # Add the cover image
    st.image("img/kakaotalk.png")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Add the "How to Use" message to the sidebar
    st.sidebar.markdown(how_to_use)

    if choice == "Home":
        st.subheader("Home")

        with st.form(key="nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

        # Layout
        col1, col2 = st.columns(2)
        if submit_button and raw_text:
            # Display balloons
            st.balloons()
            with col1:
                st.info("Results")
                # Tokenize the input text
                inputs = tokenizer([raw_text], padding=True, truncation=True, max_length=512, return_tensors='pt')

                # Make a forward pass through the model
                outputs = model(**inputs)

                # Get the predicted class and associated score
                predicted_class = outputs.logits.argmax().item()
                scores = outputs.logits.softmax(dim=1).detach().numpy()[0]

                # Mapping of prediction to sentiment labels
                sentiment_dict = {0: 'Negative', 1: 'Positive'}
                sentiment_label = sentiment_dict[predicted_class]
                confidence_level = scores[predicted_class]

                # Display sentiment
                st.write(f"Sentiment: {sentiment_label}, Confidence Level: {confidence_level:.2f}")

                # Emoji and sentiment image
                if predicted_class == 1:
                    st.markdown("Sentiment: Positive :smiley:")
                    st.image("img/positive_emoji.jpg")
                else:
                    st.markdown("Sentiment: Negative :angry:")
                    st.image("img/negative_emoji.jpg")

            # Create the results DataFrame
            results_df = pd.DataFrame({
                'Sentiment Class': ['Negative', 'Positive'],
                'Score': scores
            })

            # Create the Altair chart
            chart = alt.Chart(results_df).mark_bar(width=50).encode(
                x="Sentiment Class",
                y="Score",
                color="Sentiment Class"
            )

            # Display the chart
            with col2:
                st.altair_chart(chart, use_container_width=True)
                st.write(results_df)

    else:
        st.subheader("About")
        st.write("This is a sentiment analysis NLP app developed by Dilshod for analyzing reviews for KakakTalk mobile app on Google Play Store. It uses a fine-tuned model to predict the sentiment of the input text. The app is part of a portfolio project to showcase my nlp skills and collaboration among developers.")

if __name__ == "__main__":
    main()

