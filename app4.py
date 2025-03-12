import time
import re
import os
import csv
import json
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key="AIzaSyAZlyuffMEX-nrhfOHf5auy1SA75N7q2vs")  

CACHE_FOLDER = "transcripts"
os.makedirs(CACHE_FOLDER, exist_ok=True)

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_video_transcript(video_id):
    cache_path = os.path.join(CACHE_FOLDER, f"{video_id}.json")
    if os.path.exists(cache_path):
        print(f"📁 Using locally stored transcript for {video_id}")
        with open(cache_path, "r", encoding="utf-8") as file:
            return json.load(file)
    try:
        print(f"🌐 Fetching transcript for {video_id} from YouTube API...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(transcript, file, indent=4)

        return transcript
    except Exception as e:
        print(f"❌ Error fetching transcript for {video_id}: {e}")
        return []

def format_time(seconds):
    try:
        seconds = int(float(seconds))
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}" 
    except ValueError:
        return "00:00"

def convert_timestamp(timestamp):
    if isinstance(timestamp, str) and re.match(r"^\d+:\d{2}$", timestamp):
        return timestamp 
    return format_time(timestamp)  

def analyze_sponsorship(transcript, influencer_name, expected_product, video_url):
    transcript_formatted = "\n".join(
        f"[{format_time(entry['start'])}] {entry['text']}" for entry in transcript
    )

    prompt = f"""
You are an AI expert in advertisement analysis. Analyze the transcript of an influencer's video to extract ad details and evaluate the ad's quality.

### **Video Details:**  
- **Influencer Name:** {influencer_name}  
- **Video URL:** {video_url}
- **Transcript (with timestamps):**  
{transcript_formatted}  

### **Instructions:**  
1. **Extract the advertisement section**, including:  
   - **Exact ad text** as spoken in the video.  
   - **Start & end timestamps** (in MM:SS format).  
2. **Detect the promoted product**, including:  
   - **Product name & model (if mentioned)**  
3. **Validate against the expected product:**  
   - **Expected Product:** {expected_product}  
   - **Extracted Product:** AI-detected product  
   - **Match Accuracy:** "Yes" if extracted product matches expected product, otherwise "No".  
   - **Inference:** Explanation of why the match was successful or not.  
4. **Evaluate the advertisement quality** based on these metrics (score 1-10):  
   - **Ad Naturalness**: How smoothly the ad is integrated into the video.  
   - **Persuasiveness**: How convincing the ad is in making viewers interested.  
   - **Trustworthiness**: Does the influencer genuinely sound like they believe in the product?  
   - **Ad Length & Placement**: Was the ad length appropriate and positioned naturally?  
   - **Engagement**: Did the influencer make the ad engaging, conversational, or interactive?  
5. **Classify the advertisement as:**  
   - **Good** (8-10 average)  
   - **Moderate** (5-7 average)  
   - **Bad** (1-4 average)  

### **Expected JSON Output:**  
{{
  "influencer_name": "{influencer_name}",
  "video_url": "{video_url}",
  "advertisement_text": "Extracted ad mention",
  "product_name": "AI-detected product",
  "start_time": "MM:SS",
  "end_time": "MM:SS",
  "expected_product": "{expected_product}", 
  "match_accuracy": "Yes/No",
  "inference": "Reasoning for match or mismatch",
  "ad_naturalness": 0-10,
  "persuasiveness": 0-10,
  "trustworthiness": 0-10,
  "ad_length_placement": 0-10,
  "engagement": 0-10,
  "ad_classification": "Good/Moderate/Bad"
}}

If **no advertisement** is found, return:  
{{
  "advertisement_text": "No Advertisement Found"
}}

Ensure the response is in **valid JSON format**.
"""

    try:
        time.sleep(4) 
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        clean_text = response.text.strip().strip("```json").strip("```").strip()
        sponsorship_data = json.loads(clean_text)

        if sponsorship_data.get("advertisement_text") == "No Advertisement Found":
            return None
        sponsorship_data["start_time"] = convert_timestamp(sponsorship_data["start_time"])
        sponsorship_data["end_time"] = convert_timestamp(sponsorship_data["end_time"])

        return sponsorship_data
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON Error: {e} - LLM Response: {response.text}")
        return None
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def process_videos(video_data):
    results = []
    for index, data in enumerate(video_data):
        url = data["video_url"]
        influencer_name = data["influencer_name"]
        expected_product = data["expected_product"]

        video_id = extract_video_id(url)
        if not video_id:
            print(f"Invalid URL: {url}")
            continue

        transcript = get_video_transcript(video_id)
        if not transcript:
            print(f"No transcript available for: {url}")
            continue

        sponsorship_section = analyze_sponsorship(transcript, influencer_name, expected_product, url)
        if sponsorship_section:
            results.append(sponsorship_section)

        print(f"Processed {url}")

        if index < len(video_data) - 1:
            wait_time = 5 
            print(f"Waiting {wait_time} seconds before next request...")
            time.sleep(wait_time)

    return results

def save_results_to_csv(results, filename="sponsorship_analysis.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Influencer Name", "Video URL", "Advertisement Text", "Product", 
            "Start Time", "End Time", "Expected Product", "Match", "Inference", 
            "Ad Naturalness", "Persuasiveness", "Trustworthiness", "Ad Length & Placement", "Engagement", "Ad Classification"
        ])
        for entry in results:
            writer.writerow([
                entry['influencer_name'], entry['video_url'], entry['advertisement_text'], entry['product_name'],
                entry['start_time'], entry['end_time'], entry['expected_product'], entry['match_accuracy'], entry['inference'],
                entry['ad_naturalness'], entry['persuasiveness'], entry['trustworthiness'], 
                entry['ad_length_placement'], entry['engagement'], entry['ad_classification']
            ])
    print(f"Results saved to {filename}")
video_data = [
    {"video_url": "https://www.youtube.com/watch?v=qWMK16uYQbU", "influencer_name": "Ali Abdaal", "expected_product": "trading 212"},
    {"video_url": "https://www.youtube.com/watch?v=c_DOG_mXz5w", "influencer_name": "Ali Abdaal", "expected_product": "trading 212"},
    {"video_url": "https://www.youtube.com/watch?v=VpN78TXMSUM", "influencer_name": "Ali Abdaal", "expected_product": "BetterHelp"},
    {"video_url": "https://www.youtube.com/watch?v=Qc6pdR8BhFA", "influencer_name": "Ali Abdaal", "expected_product": "Shopify"},
    {"video_url": "https://www.youtube.com/watch?v=eJgcxYMhdoU", "influencer_name": "Ali Abdaal", "expected_product": "reclaim"},
    {"video_url": "https://www.youtube.com/watch?v=19_sGcrsWhg", "influencer_name": "The Diary Of A CEO", "expected_product": "Perfect Ted"},
    {"video_url": "https://www.youtube.com/watch?v=FjrJ2DJN_pA", "influencer_name": "The Diary Of A CEO", "expected_product": "Vanta"},
    {"video_url": "https://www.youtube.com/watch?v=eOnIWDMNyfE", "influencer_name": "The Diary Of A CEO", "expected_product": "Zoe"},
    {"video_url": "https://www.youtube.com/watch?v=rDyTyppGxSg", "influencer_name": "The Diary Of A CEO", "expected_product": "Perfect Ted"},
    {"video_url": "https://www.youtube.com/watch?v=LiIs5X56JMI", "influencer_name": "The Diary Of A CEO", "expected_product": "Linkedin"},
    {"video_url": "https://www.youtube.com/watch?v=4_5Smu0YAxw", "influencer_name": "Think School", "expected_product": "axis max life insurance"},
    {"video_url": "https://www.youtube.com/watch?v=aGVtLubW9q0", "influencer_name": "Think School", "expected_product": "GoldenPi"},
    {"video_url": "https://www.youtube.com/watch?v=I7vz7Ym82_4", "influencer_name": "Think School", "expected_product": "wind wealth"},
    {"video_url": "https://www.youtube.com/watch?v=G7Jyzj1FpnU", "influencer_name": "Think School", "expected_product": "Odoo"},
    {"video_url": "https://www.youtube.com/watch?v=FoQR9rLpRy8", "influencer_name": "Think School", "expected_product": "Scaler school of business"},
    {"video_url": "https://www.youtube.com/watch?v=Uhbf9oJ9NCs", "influencer_name": "Hyram Yarbro", "expected_product": "Stylevana"},
    {"video_url": "https://www.youtube.com/watch?v=V4GLqNOIysw", "influencer_name": "Hyram Yarbro", "expected_product": "Nira"},
    {"video_url": "https://www.youtube.com/watch?v=xKDKdIZC_j8", "influencer_name": "Hyram Yarbro", "expected_product": "iHerb"},
    {"video_url": "https://www.youtube.com/watch?v=mUtrucc4cq8", "influencer_name": "Hyram Yarbro", "expected_product": "Credo Beauty"},
    {"video_url": "https://www.youtube.com/watch?v=6zpEBAnYf4g", "influencer_name": "Hyram Yarbro", "expected_product": "Stylevana"},
    {"video_url": "https://www.youtube.com/watch?v=mcDB4pqLhfI", "influencer_name": "Bethany Mota", "expected_product": "Thredup"},
    {"video_url": "https://www.youtube.com/watch?v=yzsEnpnI-vA", "influencer_name": "Bethany Mota", "expected_product": "YouTube"},
    {"video_url": "https://www.youtube.com/watch?v=Xenj_fx4w2E", "influencer_name": "Bethany Mota", "expected_product": "Thredup"},
    {"video_url": "https://www.youtube.com/watch?v=yeWH2hxsB8Y", "influencer_name": "Bethany Mota", "expected_product": " CVS Pharmacy"},
    {"video_url": "https://www.youtube.com/watch?v=CVLEXwppll8", "influencer_name": "Bethany Mota", "expected_product": "Thredup"},
]
results = process_videos(video_data)
save_results_to_csv(results)
print("✅ Sponsorship extraction completed.")
