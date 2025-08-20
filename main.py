from object_detection import start_object_detection
from voice_alerts.voice_alerts import send_voice_alert


if __name__ == "__main__":
    print("🚀 GuardianX is starting...")
    print("Choose a mode:")
    print("1. Voice Alert Test")
    print("2. Object Detection with Voice Alerts")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        send_voice_alert("This is a test alert from GuardianX.")
    elif choice == "2":
        start_object_detection()
    else:
        print("❌ Invalid choice. Exiting...")
