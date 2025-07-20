import streamlit as st
from transformers import pipeline

# Title
st.title("Influencer Post Summarizer 🧠")
st.markdown("This app summarizes influencer posts and detects key trends.")

# User input
user_input = st.text_area("📋 Paste an Influencer Post Caption", height=200)

# Summarizer pipeline
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Button to summarize
if st.button("Summarize Post"):
    if user_input.strip() == "":
        st.warning("Please paste a post to summarize.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarizer(user_input, max_length=60, min_length=15, do_sample=False)
            st.success("🔍 Summary:")
            st.write(summary[0]['summary_text'])

# Footer
st.markdown("---")
st.caption("Made with 💻 using Hugging Face Transformers and Streamlit")