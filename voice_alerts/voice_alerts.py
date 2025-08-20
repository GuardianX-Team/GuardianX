from gtts import gTTS
import os
import playsound

def send_voice_alert(message="Emergency! Please help!"):
    try:
        # Convert text to speech
        tts = gTTS(message)
        filename = "alert.mp3"
        tts.save(filename)

        # Play the alert sound
        playsound.playsound(filename)

        print("✅ Voice alert sent successfully.")
    except Exception as e:
        print("❌ Error sending voice alert:", e)

if __name__ == "__main__":
    send_voice_alert("This is a test alert from GuardianX.")
