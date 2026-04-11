import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Ensure the parent directory is in the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.slack_api import SlackAPI, SlackAPIError
from utils.email_api import EmailAPI, EmailAPIError

def test_slack():
    print("Testing Slack integration...")
    try:
        slack_api = SlackAPI()
        response = slack_api.post_message(
            text="🚨 AutoServe AI System Test Message",
            blocks=[
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "AutoServe AI Setup Test"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "If you see this, your SLACK_BOT_TOKEN and SLACK_CHANNEL are perfectly configured! 🎉"}
                }
            ]
        )
        print(f"✅ Success! Slack message posted to channel: {response.get('channel')}\n")
    except Exception as e:
        print(f"❌ Slack Test Failed! Error: {e}\n")

def test_email():
    print("Testing Email integration...")
    try:
        email_api = EmailAPI()
        response = email_api.send_email(
            subject="AutoServe AI Configuration Test",
            text_body="If you see this, your email configuration is working!",
            html_body="<p>If you see this, your <b>email configuration</b> is perfectly configured! 🎉</p>"
        )
        print(f"✅ Success! Email sent via {response.get('provider')} to {response.get('to_email')}\n")
    except Exception as e:
        print(f"❌ Email Test Failed! Error: {e}\n")

if __name__ == "__main__":
    print("-" * 40)
    print("AutoServe AI: Marketing Infrastructure Test")
    print("-" * 40)
    
    test_slack()
    test_email()
