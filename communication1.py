import os
import re
import spacy
import contractions
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st
from textblob import TextBlob
from collections import Counter
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define filler words
FILLER_WORDS = {"uh", "um", "er", "ah", "like", "well", "right", "okay", "yeah"}
COMMON_VERBS = {"have", "do", "be", "get", "make", "go", "say", "know", "think", "see", "take"}

# Extract video ID from URL
def extract_video_id(url):
    match = re.search(r"(?:v=|\/|vi\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Fetch YouTube transcript
def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript:
            return "", 0
        transcript_text = " ".join(entry["text"] for entry in transcript)
        duration_seconds = transcript[-1]["start"] + transcript[-1].get("duration", 0)
        return transcript_text, duration_seconds
    except Exception:
        return "", 0

# Preprocess text
def preprocess_text(text):
    if not text:
        return ""
    text = contractions.fix(text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# Count words using spaCy
def count_words(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.is_alpha)

# Compute speech metrics
def compute_speech_metrics(text, duration_seconds):
    total_words = count_words(text)
    unique_words = len(set(text.split()))
    filler_count = sum(count for _, count in extract_fillers(text))
    filler_percentage = round((filler_count / total_words) * 100, 2) if total_words > 0 else 0
    speaking_pace = round(total_words / (duration_seconds / 60), 2) if duration_seconds > 0 else 0
    return total_words, unique_words, filler_percentage, speaking_pace

# Extract most used words
def extract_most_used_words(text):
    doc = nlp(text)
    words = [
        token.text for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and token.lemma_ not in COMMON_VERBS
    ]
    return Counter(words).most_common(10)

# Extract filler words
def extract_fillers(text):
    words = text.split()
    return Counter(word for word in words if word in FILLER_WORDS).most_common(5)

# Extract filler phrases
def extract_filler_phrases(text):
    words = text.split()
    two_word_phrases = Counter(" ".join(words[i:i+2]) for i in range(len(words) - 1) if words[i] in FILLER_WORDS)
    three_word_phrases = Counter(" ".join(words[i:i+3]) for i in range(len(words) - 2) if words[i] in FILLER_WORDS)
    return two_word_phrases.most_common(5), three_word_phrases.most_common(5)

# Analyze sentiment
def analyze_sentiment(text):
    if not text:
        return []
    
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    sentiment_scores = [(sent, TextBlob(sent).sentiment.polarity) for sent in sentences]
    return sentiment_scores

# Categorize sentiment
def categorize_sentiment(sentiment_scores):
    total = len(sentiment_scores)
    if total == 0:
        return {"Positive": 0, "Neutral": 0, "Negative": 0}

    positive = sum(1 for _, score in sentiment_scores if score > 0)
    neutral = sum(1 for _, score in sentiment_scores if score == 0)
    negative = sum(1 for _, score in sentiment_scores if score < 0)

    overall_sentiment = sum(score for _, score in sentiment_scores) / total
    sentiment_label = "Positive" if overall_sentiment > 0 else "Neutral" if overall_sentiment == 0 else "Negative"

    return {
        "Positive": round((positive / total) * 100, 2),
        "Neutral": round((neutral / total) * 100, 2),
        "Negative": round((negative / total) * 100, 2),
        "Overall Sentiment": overall_sentiment,
        "Label": sentiment_label
    }

# Get most positive and negative segments
def extract_sentiment_segments(sentiment_scores, top_n=2):
    sorted_scores = sorted(sentiment_scores, key=lambda x: x[1])
    most_negative = sorted_scores[:top_n]
    most_positive = sorted_scores[-top_n:]
    
    return {
        "Most Positive": [(sent, round(score, 2)) for sent, score in most_positive],
        "Most Negative": [(sent, round(score, 2)) for sent, score in most_negative]
    }
    
def extract_focused_topics(cleaned_text, top_n=10):
    if not cleaned_text:
        return []

    doc = nlp(cleaned_text)

    # Extract Named Entities (NER) - Focus on meaningful categories
    entities = [
        ent.text.lower() for ent in doc.ents 
        if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"}
    ]

    # Extract Important Noun Phrases (Filtering stopwords & pronouns)
    noun_phrases = [
        chunk.text.lower() for chunk in doc.noun_chunks
        if len(chunk.text.split()) > 1 and not any(token.is_stop or token.pos_ == "PRON" for token in chunk)
    ]

    # Combine & count frequencies
    all_topics = entities + noun_phrases
    topic_counts = Counter(all_topics)

    # Get top N topics
    return topic_counts.most_common(top_n)


# Get communication improvement suggestions
def get_gemini_suggestions(transcript_text):
    if not transcript_text:
        return "No transcript available for analysis."
    prompt = (
        "Analyze the following speech transcript and provide exactly **5 key suggestions** for improvement. "
        "Base the suggestions purely on the content, structure, and delivery of the speech. "
        "Keep each point **clear, concise, and actionable**.\n\n"
        f"Transcript:\n{transcript_text}"
    )
    response = gemini_model.generate_content(prompt)
    if response and response.text:
        suggestions = re.findall(r"^\d+\.\s.*", response.text, re.MULTILINE)
        return "\n".join(suggestions[:5]) if suggestions else response.text
    return "No suggestions generated."

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .title { font-size: 50px !important; font-weight: bold; }
        .subtitle { font-size: 35px !important; font-weight: bold; }
        .metric-label { font-size: 35px !important;}
        .metric-value { font-size: 40px !important;}
        /* Reduce font size of metric values */
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
    }

    /* Reduce font size of metric labels */
    div[data-testid="stMetricLabel"] {
        font-size: 16px !important;
    }

    /* Adjust sentiment text */
    .overall-sentiment {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ Speaker Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze speech patterns and improve communication effectiveness</p>', unsafe_allow_html=True)

# LINKS INPUT
container = st.container(border=True)  
with container:
    urls_input = st.text_area("Enter YouTube URLs (one per line):", height=150)
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]

if st.button("Analyze"):
    all_transcripts = []
    total_duration_seconds = 0  
    invalid_urls = []

    with st.spinner("Fetching transcripts..."):
        for url in urls:
            video_id = extract_video_id(url)
            if video_id:
                transcript_text, duration = get_youtube_transcript(video_id)
                if transcript_text:
                    all_transcripts.append(transcript_text)
                    total_duration_seconds += duration  
                else:
                    invalid_urls.append(url)
            else:
                invalid_urls.append(url)

    if invalid_urls:
        st.warning(f"⚠️ Invalid or missing transcripts for: {', '.join(invalid_urls)}")

    if not all_transcripts:
        st.error("❌ No valid YouTube transcripts found.")
    else:
        combined_text = preprocess_text(" ".join(all_transcripts))

        # --- Compute Speech Metrics ---
        with st.spinner("Computing speech metrics..."):
            total_words, unique_words, filler_percentage, speaking_pace = compute_speech_metrics(combined_text, total_duration_seconds)

        # --- Extract Insights ---
        with st.spinner("Extracting key insights..."):
            most_used_words = extract_most_used_words(combined_text)
            filler_words = extract_fillers(combined_text)
            two_word_fillers, three_word_fillers = extract_filler_phrases(combined_text)

        # --- Word Analysis & Sentiment Analysis ---
        container = st.container(border=True)
        container = st.container(border=True)
        with st.container():
            col1, col2 = st.columns(2)  # Two equal-width columns

            # --- Word Analysis ---
            with col1:
                st.subheader("📊 Word Analysis")
                col_w1, col_w2, col_w3, col_w4 = st.columns([1.5, 1.5, 1, 2])  # Adjusted column widths

                col_w1.metric("Total Words", total_words)
                col_w2.metric("Unique Words", unique_words)
                col_w3.metric("Filler Word %", f"{filler_percentage}%")
                col_w4.metric("Speaking Pace", f"{speaking_pace} wpm")

            # --- Sentiment Analysis ---
            with col2:
                st.subheader("📊 Sentiment Analysis")
                sentiment_scores = analyze_sentiment(combined_text)
                sentiment_results = categorize_sentiment(sentiment_scores)

                # Styled sentiment display
                st.markdown(f"""
                <p class="overall-sentiment">
                    Overall Sentiment: <span style="color:green;">{sentiment_results['Label']} ({round(sentiment_results['Overall Sentiment'], 2)})</span>
                </p>
                """, unsafe_allow_html=True)

                col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 1])  # Adjusted column widths
                col_s1.metric("😊 Positive", f"{sentiment_results['Positive']}%")
                col_s2.metric("😐 Neutral", f"{sentiment_results['Neutral']}%")
                col_s3.metric("😢 Negative", f"{sentiment_results['Negative']}%")

        # --- Graphs Section ---
        container = st.container(border=True)
        with container:
            st.subheader("📊 Word Usage & Speech Patterns")
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)

            def plot_bar_chart(data, title, color):
                df = pd.DataFrame(data, columns=["Word", "Count"])
                fig = px.bar(df, x="Word", y="Count", title=title, color="Count", color_continuous_scale=color)
                st.plotly_chart(fig, use_container_width=True)

            with col_g1:
                plot_bar_chart(most_used_words, "Most Used Words", "blues")
            with col_g2:
                plot_bar_chart(filler_words, "Filler Words", "reds")
            with col_g3:
                plot_bar_chart(two_word_fillers, "2-Word Filler Phrases", "purples")
            with col_g4:
                plot_bar_chart(three_word_fillers, "3-Word Filler Phrases", "pinkyl")


        # --- Speaker Focus Topics ---
        container = st.container(border=True)
        with st.spinner("Extracting focused topics..."):
            container = st.container(border=True)  
            with container:
                focused_topics = extract_focused_topics(preprocess_text(combined_text), top_n=5)
                st.subheader("🎯 Speaker's Focused Topics")

                st.markdown(
                        """
                        <style>
                            .circle-container {
                                display: flex;
                                flex-wrap: nowrap;
                                gap: 20px;
                                justify-content: space-evenly;
                                align-items: center;
                                margin-top: 20px;
                                overflow-x: auto;
                                white-space: nowrap;
                                padding: 10px;
                            }
                            .topic-bubble {
                                min-width: 140px;
                                height: 140px;
                                font-weight: bold;
                                font-size: 14px;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                text-align: center;
                                border-radius: 50%;
                                padding: 15px;
                                word-wrap: break-word;
                                overflow-wrap: break-word;
                                white-space: normal;
                                text-overflow: ellipsis;
                            }
                            /* Gradient background effect for 5 bubbles */
                            .topic-bubble:nth-child(1) { background: linear-gradient(to bottom, #A569BD, #D2B4DE); color: #4A148C; } /* Purple */
                            .topic-bubble:nth-child(2) { background: linear-gradient(to bottom, #5499C7, #AED6F1); color: #1A5276; } /* Blue */
                            .topic-bubble:nth-child(3) { background: linear-gradient(to bottom, #48C9B0, #A2D9CE); color: #0B5345; } /* Green */
                            .topic-bubble:nth-child(4) { background: linear-gradient(to bottom, #F5B041, #FAD7A0); color: #935116; } /* Orange */
                            .topic-bubble:nth-child(5) { background: linear-gradient(to bottom, #EC7063, #F5B7B1); color: #7B241C; } /* Red */
                        </style>
                        """,
                        unsafe_allow_html=True
                    )


            topic_html = '<div class="circle-container">'
            for topic, _ in focused_topics:
                topic_html += f'<div class="topic-bubble">{topic}</div>'
            topic_html += "</div>"
            st.markdown(topic_html, unsafe_allow_html=True)

        # --- Suggestions/Analysis Comments ---
        container = st.container(border=True)
        with container:
            st.subheader("📌 Suggestions & Analysis Comments")
            suggestions = get_gemini_suggestions(combined_text)
            st.write(suggestions)
