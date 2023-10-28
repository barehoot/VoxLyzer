import streamlit as st
import plotly.graph_objects as go
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import os
import re
import nltk
import pandas as pd
import docx
import PyPDF2
import textract
from PyPDF2 import PdfFileReader
from io import BytesIO
from PIL import Image
import fitz
import warnings
from PyPDF2 import PdfReader
import speech_recognition as sr
from pydub import AudioSegment
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('words')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Text Sentiment Analysis", page_icon=":1234:", layout= 'wide')


header_animation_css = """
<style>
.header-animation {
    animation: slideUp 0.5s forwards;
}
@keyframes slideUp {
    from {
        transform: translateY(0);
        opacity: 1;
    }
    to {
        transform: translateY(-100%);
        opacity: 0;
    }
}
</style>
"""
st.markdown(
    """
    <style>
    .sidebar-content .sidebar-collapser .icon {
        width: 40px; /* Adjust the width to change the size */
        height: 40px; /* Adjust the height to change the size */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Page title style */
    .title {
        font-size: 24px;
        text-align: center;
    }
    /* Text styling */
    .text {
        color: #333;
        font-size: 18px;
    }
    .title2 {
    font-size: 24px;
    text-align: center; /* Horizontal centering */
    margin-top: -10px; /* Adjust the value to move the text up */
    /* Add additional styles as needed */
}

    </style>
    """,
    unsafe_allow_html=True,
)

css_styles = """
<style>
    /* Page title style */
    .title {
        font-size: 24px;
        text-align: center;
        color: #333;
    }

    /* Subtitle style */
    .subtitle {
        font-size: 18px;
        color: #6c736e;
    }

    /* Centered elements */
    .center {
        text-align: center;
    }

    /* Speedometer chart container */
    .speedometer-chart {
        border: 2px solid #ddd;
        padding: 50px;
        border-radius: 10px;
    }

    /* Word Cloud container */
    .word-cloud {
        border: 2px solid #ddd;
        padding: 10px;
        border-radius: 10px;
    }

    /* Text editor */
    .text-editor {
        border: 2px solid #ddd;
        border-radius: 10px;
    }

    /* Button style */
    .custom-button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
"""

st.markdown(css_styles, unsafe_allow_html=True)


st.markdown(header_animation_css, unsafe_allow_html=True)  # Apply the animation CSS
st.markdown('<h1 class="title">Text Sentiment Analysis Data Tool</h1>', unsafe_allow_html=True)
st.markdown('<center><h5 style="color: grey;">This Tool scrape the Article from the url and Analyse the uploaded DOCX, PDF and TXT File</h5></center>', unsafe_allow_html=True)
st.markdown('<center><h5 style="color: grey;" >Note : Please got to the the Sidebar/Navbar to enter the link or upload the file- to initialize the Tool</h5></center>', unsafe_allow_html=True)

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* Change button background color */
    .stButton > button {
        background-color: #90EEFR;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tokenization and Preprocessing
def tokenize_and_preprocess(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# TF-IDF Vectorization
def tfidf_vectorize(text_data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_vectorizer, tfidf_matrix

# Calculate TF-IDF Scores
def calculate_tfidf_scores(query, tfidf_vectorizer, tfidf_matrix):
    query = tokenize_and_preprocess(query)
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix)
    return cosine_similarities

# Retrieve Relevant Information
def retrieve_relevant_information(text_data, query, cosine_similarities):
    scores = list(cosine_similarities[0])
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_sentences = [text_data[i] for i in sorted_indices][:5]  # Get the top 5 relevant sentences
    return top_sentences


def transcribe_audio(uploaded_audio):
    if uploaded_audio:
        recognizer = sr.Recognizer()

        with st.spinner("Transcribing..."):
            # Convert any audio format to WAV using pydub
            audio = AudioSegment.from_file(uploaded_audio)
            wav_filename = "audio.wav"
            audio.export(wav_filename, format="wav")

            # Transcribe the WAV file
            audio_data = sr.AudioFile(wav_filename)
            with audio_data as source:
                audio_text = recognizer.record(source)

            text = recognizer.recognize_google(audio_text)
            st.success("Transcription Complete:")
            sent(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()  # Split the text into words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def perform_pos_tagging(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return pos_tags


def create_speedometer_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "lightgray"},
            'steps': [
                {'range': [0.66, 1], 'color': "green"},
                {'range': [0.33, 0.66], 'color': "yellow"},
                {'range': [0, 0.33], 'color': "red"}
            ],
        }
    ))

    fig.update_layout(width=300, height=500)
    return fig



def create_speedometer_chart2(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "lightgray"},
            'steps': [
                {'range': [0.4, 1], 'color': "green"},
                {'range': [-0.5, 0.4], 'color': "yellow"},
                {'range': [-1, -0.5], 'color': "red"}
            ],
        }
    ))
    fig.update_layout(width=300, height=500)

    return fig

def generate_wordcloud(content):
    # Create a WordCloud object
    #Create a WordCloud object
    # BG COL: #181D35
    wordcloud = WordCloud(width=1500, height=800,
                          background_color='#EFECEC',
                          colormap='viridis',
                          min_font_size=10).generate(content)
    
    st.markdown(
        f'<style>img.stImage {{ background-color: transparent; }}</style>',
        unsafe_allow_html=True)
    # Display the Word Cloud using Matplotlib
    st.image(wordcloud.to_array())


def text_sent(content):
    sia = SentimentIntensityAnalyzer()
    # Remove stopwords from the content
    content_without_stopwords = remove_stopwords(content)
    sentiment = sia.polarity_scores(content_without_stopwords)
    return sentiment


def read_txt(upl):
    content = upl.read().decode("utf-8")
    return content

def read_pdf(upl):
    pdf_reader = PdfReader(upl)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    return content


def read_docx(upl):
    doc = docx.Document(upl)
    content = []

    for paragraph in doc.paragraphs:
        content.append(paragraph.text)

    return '\n'.join(content)

def sent(content):

    unfiltered=content

    # Perform sentiment analysis
    sentiment = text_sent(content)
    # Create multiple columns for side-by-side display
    st.markdown(
    """
    <style>
    /* Increase the text size for the tab labels */
    .stTabs .sTtab {
        font-size: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Indicator", "Word Cloud", "Content", "Information Retrieval and Relevance Ranking"])
    

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
    # Display speedometer charts in separate columns
        with col1:
            st.plotly_chart(create_speedometer_chart(sentiment['pos'], "Positive Score"))
        with col2:
            st.plotly_chart(create_speedometer_chart(sentiment['neg'], "Negative Score"))
        with col3:
            st.plotly_chart(create_speedometer_chart(sentiment['neu'], "Neutral Score"))
        with col4:
            st.plotly_chart(create_speedometer_chart2(sentiment['compound'], "Sentiment Score"))
    with tab2:
        st.markdown('<center style="title2"><h5> Word Cloud</h5></center>', unsafe_allow_html=True)
        generate_wordcloud(unfiltered)
    with tab3:
        st.markdown('<center><h4> Read the Content Below</h4></center>', unsafe_allow_html=True)
    #st.text(content)  # Display the content
        text_area = st.empty()
    # Create an embedded scrollbar text area
        text = text_area.text_area("Edit and Scroll Text", unfiltered, height=400)
    # Create a separate section for POS tagging results
        st.markdown('<center><h4> Part-of-Speech Tagging</h4></center>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        if col1.button("Perform POS Tagging"):
            pos_tags = perform_pos_tagging(unfiltered)
            pos_tags_str = '\n'.join([f'{word}: {tag}' for word, tag in pos_tags])
            st.text_area("Part-of-Speech Tags", pos_tags_str)
            txt_content = pos_tags_str.encode('utf-8')
            st.download_button(
        "Download Data",
        data=txt_content,
        file_name="POStag.txt",
        key="pos_tag_download",
        mime="text/plain",  # Specify MIME type for a text file
        help='Click here to download the data as a txt file'
    )
    with tab4:
        try:
            query = st.text_input("Enter your query:")
            if st.button("Search in the text"):
                if not query:
                    st.error("Enter the query; it cannot be empty")
                else:
                    # Tokenize and preprocess the content
                    content = tokenize_and_preprocess(content)
                    content_data = [content]

                    # TF-IDF Vectorization
                    tfidf_vectorizer, tfidf_matrix = tfidf_vectorize(content_data)

                    # Calculate TF-IDF Scores
                    cosine_similarities = calculate_tfidf_scores(query, tfidf_vectorizer, tfidf_matrix)

                    # Retrieve Relevant Information
                    relevant_info = retrieve_relevant_information(content_data, query, cosine_similarities)

                    st.subheader("Top Relevant Sentences:")
                    for i, sentence in enumerate(relevant_info, start=1):
                        st.write(f"{i}. {sentence}")

                    # Create a downloadable text file
                    download_text = "\n".join(relevant_info)
                    st.download_button(
                    "Download Relevant Sentences",
                    download_text.encode("utf-8"),
                    key="download_relevant_sentences",
                    mime="text/plain",
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def text_analysis(url):
    try:
            # Initialize Article object
            article = Article(url)

            # Download and parse the article
            article.download()
            content = article.parse()
            content = article.text  # Extract the article text
            sent(content)
            

    except Exception as e:
        st.error(f"An error occurred while processing URL {url}: {str(e)}")

st.sidebar.header("Enter the News Article Link or upload the file")
url = st.sidebar.text_input("Input the Newsarticle URL link")
if st.sidebar.button("Submit"):
    if url:
        text_analysis(url)
    else:
        st.warning("Enter the URL: it cannot be empty")
upl= st.sidebar.file_uploader("Upload file", type=(["docx","txt","pdf"]))
uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["mp3", "wav", "flac"])



#if upl is not None:
    #text_content = upl.read()
  #  sent(read_docx(upl))

if st.sidebar.button("Transcribe"):
    transcribe_audio(uploaded_audio)
    st.write("Currently Audio Function is not working properly, woorking on it get it resolved. Soon!! Thanks for your support.")
    if transcribe_audio:
        st.audio(uploaded_audio, format="audio/wav")  

if st.sidebar.button("Clear All"):
    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
    st.cache_data.clear()
    

if upl is not None:
    file_extension = upl.name.split('.')[-1].lower()

    if file_extension == "docx":
        content = read_docx(upl)
    elif file_extension == "txt":
        content = read_txt(upl)
    elif file_extension == "pdf":
        content = read_pdf(upl)
    else:
        st.warning("Unsupported file format. Please upload a DOCX, TXT, or PDF file.")

    if 'content' in locals():
        sent(content)  # Perform sentiment analysis and display results


