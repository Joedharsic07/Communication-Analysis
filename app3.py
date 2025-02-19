import os
import re
import spacy
import contractions
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_video_id(url):
    """Extracts YouTube video ID from a given URL."""
    match = re.search(r"(?:v=|\/|vi\/|youtu\.be\/|embed\/|shorts\/|watch\?v=|&v=)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_youtube_transcript(video_url):
    """Fetches YouTube transcript with timestamps."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return []
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [(entry['start'], entry['duration'], entry['text']) for entry in transcript]
    except TranscriptsDisabled:
        print(f"⚠ Transcript is disabled for video: {video_url}")
    except Exception as e:
        print(f"❌ Error fetching transcript for {video_url}: {str(e)}")
    return []

def preprocess_text_with_timestamps(transcript):
    """Cleans transcript while keeping timestamps."""
    processed_transcript = []
    for start, duration, text in transcript:
        processed_text = contractions.fix(text.lower())  # Expand contractions and lowercase
        processed_text = re.sub(r'[\[\](){}]', '', processed_text)  # Remove brackets
        doc = nlp(processed_text)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        cleaned_text = " ".join(words)
        processed_transcript.append((start, duration, cleaned_text))
    return processed_transcript

def truncate_transcript(cleaned_transcript, max_words=5000):
    """Ensures transcript doesn’t exceed Gemini's token limit."""
    transcript_text = " ".join([text for _, _, text in cleaned_transcript])
    words = transcript_text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])  # Truncate to avoid exceeding Gemini’s limit
    return transcript_text

def analyze_with_gemini(cleaned_transcript):
    """Passes the cleaned transcript to Gemini for analysis."""
    transcript_text = truncate_transcript(cleaned_transcript)

    prompt = f"""
## 📊 YouTube Transcript Analysis Report  
You are an **expert linguistic analyzer**. Your job is to **strictly extract and categorize** words found in the transcript.  

### **⚠ Important Rules**  
✅ **Only extract words that actually exist in the transcript.**  
✅ **Do NOT assume or predict words—analyze only what is provided.**  
✅ **If any word is mistakenly classified as a filler, move it to the Non-Filler Words section.**  

---

### **1️⃣ Most Frequently Used Words**  
- List the **top 10 most used meaningful words** (excluding stopwords and fillers).  

---

### **2️⃣ Actual Filler Words**  
- Extract only **true filler words**:  
  - Examples: **"uh", "um", "like", "so", "okay", "you know", "I mean"**.  
  - If a word is mistakenly classified as a filler, move it to **Non-Filler Words** instead.  

---

### **3️⃣ Non-Filler Words**  
- Extract and rank the **top 20 meaningful words**.  
- If any word was **wrongly classified as a filler**, move it here instead.  

---

### **4️⃣ Common 2-Word Filler Phrases**  
- Extract **2-word filler phrases** (e.g., "um like", "you know").  

---

### **5️⃣ Common 3-Word Filler Phrases**  
- Extract **3-word filler phrases** (e.g., "uh you know", "um like that").  

---

### 📌 **Transcript for Analysis:**  
{transcript_text}  
"""

    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.0, "top_p": 0.5, "top_k": 1})
        return response.text
    except Exception as e:
        return f"❌ Error analyzing with Gemini: {str(e)}"


# List of YouTube videos
video_urls = [
    "https://youtu.be/2MkNdJXXdaQ?si=hKUxBSqyWbp6JXpQ",
    "https://youtu.be/metQ-DtQISo?si=a_o2H1EwKBc3lc-Y",
    "https://youtu.be/ghfITyxZcs4?si=E7cpYEoeokNVn5b1",
    "https://youtu.be/3OvmwM61vJw?si=545tgrgVPD-J_ncy",
    "https://youtu.be/un0SjUnHvvE?si=eCuVhgNepPs1FBdS"
]

# Fetch transcripts concurrently
with ThreadPoolExecutor() as executor:
    transcripts = list(executor.map(get_youtube_transcript, video_urls))

# Combine and clean transcripts
combined_transcript = [entry for sublist in transcripts for entry in sublist if sublist]

if combined_transcript:
    cleaned_transcript = preprocess_text_with_timestamps(combined_transcript)
    gemini_analysis = analyze_with_gemini(cleaned_transcript)

    print("\n📊 **Final Gemini Analysis:**")
    print(gemini_analysis if gemini_analysis else "Gemini analysis returned no output.")
else:
    print("No valid transcripts found.")
