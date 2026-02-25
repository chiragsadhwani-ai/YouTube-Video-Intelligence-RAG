import re
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    pattern = r'(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_raw_transcript(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        print(f"[ERROR] Could not extract video ID from: {url}")
        return None
    try:
        ytt = YouTubeTranscriptApi()          # instantiate first
        transcript = ytt.fetch(video_id)      # then call .fetch()
        return transcript
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return None

def format_transcript_content(transcript_data) -> str:
    return " ".join([entry.text for entry in transcript_data])  # .text not ['text']