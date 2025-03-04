from youtube_transcript_api import YouTubeTranscriptApi
import re
import csv
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyB20z6vCS78O4ro9eEmkEpFb9KJGOH-ws8")

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [(entry['start'], entry['text']) for entry in transcript]
    except Exception:
        return []

def format_time(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes:02}:{seconds:02}"

def detect_ad_segments(transcript):
    ad_keywords = [
        "sponsored by", "brought to you by", "today's sponsor", "use code", 
        "partnered with", "in collaboration with", "this video is sponsored",
        "our sponsor", "this video is made possible by", "exclusive offer", "special discount",
        "click the link", "limited time offer", "visit our partner", "get yours now", "sponsoring this video"
    ]
    
    weak_keywords = ["check out", "special deal", "discount"]
    
    ad_segments = []
    in_ad = False
    ad_start = None
    ad_texts = []
    
    for i, (start_time, sentence) in enumerate(transcript):
        lower_sentence = sentence.lower()
        keyword_matches = sum(1 for keyword in ad_keywords if keyword in lower_sentence)
        weak_matches = sum(1 for keyword in weak_keywords if keyword in lower_sentence)
        
        if keyword_matches >= 1 or (keyword_matches + weak_matches) >= 2:
            if not in_ad:
                ad_start = start_time
                ad_texts = []
                backtrack = max(0, i - 5)
                while backtrack < i and len(transcript[backtrack][1].split()) < 20:
                    ad_start = transcript[backtrack][0]
                    ad_texts.insert(0, transcript[backtrack][1])
                    backtrack += 1
                in_ad = True
            ad_texts.append(sentence)
        else:
            if in_ad:
                forward_check = i + 1
                non_ad_count = 0
                
                while forward_check < len(transcript) and non_ad_count < 2:
                    next_sentence = transcript[forward_check][1]
                    if any(keyword in next_sentence.lower() for keyword in ad_keywords):
                        ad_texts.append(next_sentence)
                        non_ad_count = 0  
                    else:
                        ad_texts.append(next_sentence)
                        non_ad_count += 1 
                    forward_check += 1
                
                ad_segments.append((ad_start, transcript[min(forward_check, len(transcript) - 1)][0], " ".join(ad_texts)))
                in_ad = False
    
    return ad_segments

def extract_product_name_gemini(ad_text):
    prompt = f"""
    Identify the main product, brand, or sponsor being promoted in the following advertisement text:
    "{ad_text}"
    
    Return only the name of the product, brand, or sponsor without additional explanations.
    If no clear product or brand is mentioned, return 'Unknown'.
    """
    
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text.strip() if response else "Unknown"

def process_videos(video_urls):
    results = []
    for url in video_urls:
        video_id = extract_video_id(url)
        if not video_id:
            print(f"Invalid URL: {url}")
            continue
        
        transcript = get_video_transcript(video_id)
        if not transcript:
            print(f"No transcript available for: {url}")
            continue
        
        ad_segments = detect_ad_segments(transcript)
        for start_time, end_time, ad_text in ad_segments:
            product_name = extract_product_name_gemini(ad_text)
            results.append({
                "video_url": url,
                "start_time": format_time(start_time),
                "end_time": format_time(end_time),
                "ad_text": ad_text,
                "product": product_name
            })
        
        print(f"Processed {url}")
    return results

def save_results_to_csv(results, filename="ad_experience_analysis.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Video URL", "Start Time", "End Time", "Ad Text", "Product"])
        for entry in results:
            writer.writerow([entry['video_url'], entry['start_time'], entry['end_time'], entry['ad_text'], entry['product']])
    print(f"Results saved to {filename}")

youtube_urls = [
    "https://www.youtube.com/watch?v=qWMK16uYQbU",
    "https://www.youtube.com/watch?v=c_DOG_mXz5w",
    "https://www.youtube.com/watch?v=VpN78TXMSUM",
    "https://www.youtube.com/watch?v=Qc6pdR8BhFA",
    "https://www.youtube.com/watch?v=eJgcxYMhdoU",
    "https://www.youtube.com/watch?v=NWfIrmIgaCU",
    "https://www.youtube.com/watch?v=RlzV8EnEwc0&t=416s",
    "https://www.youtube.com/watch?v=U7JNdMbj1zM",
    "https://www.youtube.com/watch?v=qh75NllzOlU",
    "https://www.youtube.com/watch?v=dPOX5SxcWUo",
]

results = process_videos(youtube_urls)
save_results_to_csv(results)
print("Ad detection completed.")
