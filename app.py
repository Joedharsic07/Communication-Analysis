import spacy
from collections import Counter
import re
from youtube_transcript_api import YouTubeTranscriptApi
import contractions

nlp = spacy.load("en_core_web_trf")
FILLER_WORDS = {"um", "uh", "like", "you know", "so", "well", "actually", "basically", "literally", "right", "okay", "hmm", "yeah", "kind of", "sort of"}

# extract video ID from YouTube link
def extract_video_id(video_url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    return match.group(1) if match else None

# get transcript from youtube
def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format.")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])
    return transcript_text

# clean and process the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = contractions.fix(text)
    doc = nlp(text)
    words = []
    fillers = []
    for token in doc:
        if token.text in FILLER_WORDS:
            fillers.append(token.text)
        elif not token.is_stop and not token.is_punct and len(token.text) > 1:
            words.append(token.text)
    return words, fillers

# get most frequently used words

def get_most_frequent_words(text, top_n=10):
    words, _ = preprocess_text(text)
    return Counter(words).most_common(top_n)

def get_filler_words(text):
    _, fillers = preprocess_text(text)
    return Counter(fillers).most_common()

# Load the YouTube video
video_url = "https://youtu.be/Y_uOqhhm_8s?si=XpNrrh9JZhj6qTlu"
transcript = get_youtube_transcript(video_url)
filler_words = get_filler_words(transcript)
# get most used words
most_used_words = get_most_frequent_words(transcript, top_n=10)
print("\nTranscript Preview:\n", transcript[:300])
print("\nMost Used Words:", most_used_words)
print("\nFiller Words:", filler_words)
