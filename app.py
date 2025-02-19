import os
import re
import contractions
import spacy
from collections import Counter
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define common filler words
FILLER_WORDS = {"uh", "um", "er", "ah", "like", "well", "right", "okay", "yeah"}
COMMON_VERBS = {"have", "do", "be", "get", "make", "go", "say", "know", "think", "see", "take"}

# Extract video ID from URL
def extract_video_id(url):
    match = re.search(r"(?:v=|\/|vi\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Fetch YouTube transcript
def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return ""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(entry["text"] for entry in transcript)
    except Exception:
        return ""

# Preprocess text: Expand contractions & clean punctuation
def preprocess_text(text):
    text = contractions.fix(text.lower())
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text.strip()

# Extract meaningful words using spaCy
def extract_meaningful_words(text):
    doc = nlp(text)
    words = [
        token.text
        for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and token.lemma_ not in COMMON_VERBS
    ]
    return Counter(words).most_common(10)

# Extract hesitation-based fillers
def extract_fillers(text):
    words = text.split()
    filler_counts = Counter(word for word in words if word in FILLER_WORDS)
    return filler_counts.most_common(10)

# Extract 2-word and 3-word filler phrases
def extract_filler_phrases(text):
    two_word_phrases = Counter()
    three_word_phrases = Counter()
    
    words = text.split()
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if words[i] in FILLER_WORDS or words[i+1] in FILLER_WORDS:
            two_word_phrases[bigram] += 1

    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        if words[i] in FILLER_WORDS or words[i+1] in FILLER_WORDS or words[i+2] in FILLER_WORDS:
            three_word_phrases[trigram] += 1

    return two_word_phrases.most_common(10), three_word_phrases.most_common(10)

video_urls = [
    "https://youtu.be/2MkNdJXXdaQ?si=hKUxBSqyWbp6JXpQ",
    "https://youtu.be/metQ-DtQISo?si=a_o2H1EwKBc3lc-Y",
    "https://youtu.be/ghfITyxZcs4?si=E7cpYEoeokNVn5b1",
    "https://youtu.be/3OvmwM61vJw?si=545tgrgVPD-J_ncy",
    "https://youtu.be/un0SjUnHvvE?si=eCuVhgNepPs1FBdS"
]

# Process all transcripts
transcripts = [preprocess_text(get_youtube_transcript(url)) for url in video_urls if get_youtube_transcript(url)]
combined_transcript = " ".join(transcripts) if transcripts else ""

# Perform NLP analysis
if combined_transcript:
    meaningful_words = extract_meaningful_words(combined_transcript)
    filler_counts = extract_fillers(combined_transcript)
    two_word_fillers, three_word_fillers = extract_filler_phrases(combined_transcript)

    # Print results
    print("\nMost Used Words:")
    for word, count in meaningful_words:
        print(f"{word} ({count})")

    print("\nMost Common 1-Word Fillers:")
    for word, count in filler_counts:
        print(f"{word} ({count})")

    print("\nMost Common 2-Word Filler Phrases:")
    for phrase, count in two_word_fillers:
        print(f"{phrase} ({count})")

    print("\nMost Common 3-Word Filler Phrases:")
    for phrase, count in three_word_fillers:
        print(f"{phrase} ({count})")
else:
    print("No valid transcripts found.")
