import os
import re
import spacy
import contractions
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

nlp = spacy.load("en_core_web_sm")

def extract_video_id(url):
    """Extracts YouTube video ID from a given URL."""
    match = re.search(r"(?:v=|\/|vi\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_youtube_transcript(video_url):
    """Fetches YouTube transcript with timestamps."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return []
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [entry['text'] for entry in transcript]
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        return []

def preprocess_text(transcript):
    """Cleans the transcript and removes unnecessary characters."""
    processed_transcript = []
    
    for text in transcript: 
        processed_text = contractions.fix(text.lower())  
        processed_text = re.sub(r'[\[\](){}]', '', processed_text) 
        doc = nlp(processed_text)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        cleaned_text = " ".join(words)
        processed_transcript.append(cleaned_text) 

    return processed_transcript



def analyze_with_gemini(cleaned_transcript):
    """Passes the cleaned transcript to Gemini for analysis."""
    transcript_text = " ".join(cleaned_transcript)

    prompt = f"""
## 📊 YouTube Transcript Analysis Report  
You are a linguestic expert help me to find the required words in the transcript which i pass
You are given a **cleaned transcript** from multiple YouTube videos.  
Your task is to dynamically analyze the text and generate insights **only from the transcript, without assuming words**.  

### **Important:**  
✔ **Do NOT generate words based on expectation.**  
✔ **Only count words that are actually in the transcript.**  
✔ **No hallucination—your analysis must be 100% based on the transcript.**  

---

## **Your Analysis Should Include:**  

### **1️⃣ Most Frequently Used Words**  
- Identify the **most frequently used meaningful words**.  
- **Exclude filler words** (e.g., "uh", "um", "like", "so", "okay").  
- **Exclude stopwords** (e.g., "the", "is", "are", "it", "and", "on", "in", "to", etc.).  
- **Strictly extract words from the transcript**—do not assume common words.  
- Provide a **ranked list with exact counts** of the **top 10 most meaningful words**.  

---

### **2️⃣ Actual Filler Words**  
- Identify **only real filler words**—words that frequently appear but do not add meaning.  
- **Strictly exclude the following words from being classified as fillers:**  
  - "probably," "actually," "basically," "specifically," "definitely," "really."  
- **Only include words that are truly filler, such as "uh," "um," "like," "so," "okay," etc.**  
- Provide a **ranked list of only valid filler words** found in the transcript.
- If non-filler words appear hear just move them to the non-filler words section.  

---

### **3️⃣ Non-Filler Words**  
- **Identify and rank the top 20 most meaningful non-filler words.**  
- **Only include words that provide strong contextual meaning**.  
- **Strictly exclude common words** that are not true non-fillers.  
- Words like **"probably," "actually," "basically," "specifically," "definitely," and "really" must be included.**  
- Only extract the word which is available in the transcript
---

### **4️⃣ Common 2-Word Filler Phrases**  
- Identify **frequently occurring sequences of two consecutive filler words**.  
- **Only detect actual filler word pairs** like "uh you" or "um like."  
- **Ensure these sequences exist in the transcript—do not assume them.**  
- Provide a **ranked list with exact counts** of the most common **2-word filler phrases**.  

---

### **5️⃣ Common 3-Word Filler Phrases**  
- Identify **frequently occurring sequences of three consecutive filler words**.  
- **Only detect valid filler phrases** such as "uh you know" or "um like that."  
- **Ensure these phrases exist in the transcript—do not assume them.**  
- Provide a **ranked list with exact counts** of the most common **3-word filler phrases**.  

---

### **Strict Rules for Analysis:**  
✅ **All detected words and phrases must come directly from the transcript.**  
✅ **Do not generate words based on expectation—only report what is truly found.**  
✅ **Cross-check word counts before including them.**  
✅ **Output must be structured clearly** with headings and ranked lists.  
✅ **Provide exact counts** for every detected word and phrase.  

---

### **📌 Transcript for Analysis:**  

{transcript_text}  

Now, strictly analyze the transcript and generate the final report.
"""

    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.0, "top_p": 0.5, "top_k": 1})
        return response.text
    except Exception as e:
        return f"Error analyzing with Gemini: {str(e)}"

video_urls = [
    "https://youtu.be/2MkNdJXXdaQ?si=hKUxBSqyWbp6JXpQ",
    "https://youtu.be/metQ-DtQISo?si=a_o2H1EwKBc3lc-Y",
    "https://youtu.be/ghfITyxZcs4?si=E7cpYEoeokNVn5b1",
    "https://youtu.be/3OvmwM61vJw?si=545tgrgVPD-J_ncy",
    "https://youtu.be/un0SjUnHvvE?si=eCuVhgNepPs1FBdS"
]

with ThreadPoolExecutor() as executor:
    transcripts = list(executor.map(get_youtube_transcript, video_urls))

combined_transcript = [entry for sublist in transcripts for entry in sublist] if transcripts else []

if combined_transcript:
    cleaned_transcript = preprocess_text(combined_transcript)
    transcript_text = " ".join(cleaned_transcript)  

    gemini_analysis = analyze_with_gemini(cleaned_transcript)
    print("\n📊 **Final Gemini Analysis:**")
    print(gemini_analysis if gemini_analysis else "Gemini analysis returned no output.")
else:
    print("No valid transcripts found.")