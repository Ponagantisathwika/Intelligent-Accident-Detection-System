import smtplib
from email.message import EmailMessage
from twilio.rest import Client
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Twilio Configuration from Environment Variables
TWILIO_SID = os.getenv("TWILIO_SID", "AC77f58951871839c6c9d6d525f03b91e2")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "bc52f463565e8478a72819d51fb58e10")
TWILIO_PHONE = os.getenv("TWILIO_PHONE", "+14482304825")         # Your Twilio phone number
RECIPIENT_PHONE = os.getenv("RECIPIENT_PHONE", "+91 94944 58553")   # Recipient's phone number

# Email Configuration from Environment Variables
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "sreeja1028.domakuntla@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "knrr xqwt wymp sfdr")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "sivaprasad.chebolu@gmail.com")

def send_email_alert(subject, body, image_path=None):
    try:
        # ✅ Multiple recipients
        recipients = [
            "ponagantisathwika@gmail.com",
            "jessyrider1411@gmail.com",
            "ishanigujjula@gmail.com",
            "police@gmail.com"
        ]

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        # ✅ Optional CC / BCC (uncomment if needed)
        # msg["Cc"] = "controlroom@gmail.com"
        # msg["Bcc"] = "admin@gmail.com"

        # ✅ Attach snapshot if available
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
                file_name = os.path.basename(image_path)

            msg.add_attachment(
                img_data,
                maintype="image",
                subtype="jpeg",
                filename=file_name
            )
            logging.info(f"📎 Attached image: {file_name}")

        # ✅ Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        logging.info("✅ Email sent successfully to multiple recipients.")

    except Exception as e:
        logging.error(f"❌ Failed to send email alert: {e}")

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content(body)

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
                file_name = os.path.basename(image_path)
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=file_name)
            logging.info(f"📎 Attached image: {file_name}")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            logging.info("✅ Email sent successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to send email: {e}")

def send_sms_alert(message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_PHONE, to=RECIPIENT_PHONE)
        logging.info("✅ SMS sent successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to send SMS: {e}")
