from gtts import gTTS
import playsound
import os

def send_voice_alert(message="Emergency! Please help!"):
    try:
        tts = gTTS(message)
        filename = "alert.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"Voice alert error: {e}")
