def process_chat_command(text):
    print(f"COMMAND RECEIVED: {text}")
    if "random dance" in text.lower():
        return "RANDOM_DANCE"
    return None
