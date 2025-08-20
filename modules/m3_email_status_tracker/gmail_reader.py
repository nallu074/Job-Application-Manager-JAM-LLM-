import os
import base64
import datetime as dt
from email.utils import parsedate_to_datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scope to read Gmail inbox
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def authenticate_gmail():
    creds = None
    token_path = "token.json"
    credentials_path = os.getenv("GMAIL_CREDENTIALS", "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def search_emails(service, query, after_date):
    results = []

    after_epoch = int(dt.datetime.combine(after_date, dt.datetime.min.time()).timestamp())
    full_query = f"{query} after:{after_epoch}"

    response = service.users().messages().list(userId='me', q=full_query, maxResults=10).execute()
    messages = response.get('messages', [])

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = msg_data.get("payload", {}).get("headers", [])

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(Unknown)")
        date_header = next((h["value"] for h in headers if h["name"] == "Date"), "")
        snippet = msg_data.get("snippet", "")

        try:
            parsed_date = parsedate_to_datetime(date_header)
        except Exception:
            parsed_date = None

        results.append({
            "subject": subject,
            "from": sender,
            "date": parsed_date,
            "snippet": snippet
        })

    return results

def classify_email_status(subject: str, snippet: str) -> str:
    content = f"{subject} {snippet}".lower()

    if any(kw in content for kw in ["invited", "interview", "assessment", "call with", "schedule a call", "meet with", "shortlisted"]):
        return "Interview"
    elif any(kw in content for kw in ["unfortunately", "not selected", "regret", "rejected", "declined"]):
        return "Rejected"
    else:
        return "No Response"