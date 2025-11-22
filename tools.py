
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# -----------------------------
# 1. AUDIO TRANSCRIPTION
# -----------------------------
def transcribe_audio(path):
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    return resp.text

# -----------------------------
# 2. GOOGLE CALENDAR
# -----------------------------
def get_calendar_events(query="today"):
    return "Fetched Google Calendar events"

def create_calendar_event(event):
    return f"Event created: {event}"

# -----------------------------
# 3. GMAIL
# -----------------------------
def gmail_get_emails():
    return "Here are your emails"

def gmail_send_email(data):
    return f"Email sent: {data}"

# -----------------------------
# 4. TODOIST
# -----------------------------
def todoist_get_tasks():
    return "Here are your Todoist tasks"

def todoist_create_task(task):
    return f"Task created: {task}"

# -----------------------------
# 5. NOTES
# -----------------------------
def create_note(text):
    return f"Note created: {text}"

# -----------------------------
# 6. IMAGE (FLUX)
# -----------------------------
def create_flux_image(prompt):
    response = client.images.generate(
        model="flux",
        prompt=prompt
    )
    return response.data[0].url
