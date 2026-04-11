from pipeline.analyzer import ConversationAnalyzer

# Create analyzer
analyzer = ConversationAnalyzer()

# Fake conversation (REALISTIC + MIXED)
conversation = [
    {"speaker": "User1", "cleaned_message": "Hey, how are you doing today?", "timestamp": "10:00"},
    {"speaker": "User2", "cleaned_message": "I’m fine, thanks for asking", "timestamp": "10:01"},

    # Neutral
    {"speaker": "User1", "cleaned_message": "Okay noted", "timestamp": "10:02"},

    # Sarcasm
    {"speaker": "User2", "cleaned_message": "Oh great, exactly what I needed", "timestamp": "10:03"},

    # Manipulation
    {"speaker": "User1", "cleaned_message": "If you actually cared, you would help me", "timestamp": "10:04"},

    # Genuine
    {"speaker": "User2", "cleaned_message": "I really appreciate you being honest", "timestamp": "10:05"},

    # Subtle sarcasm
    {"speaker": "User1", "cleaned_message": "Yeah because that worked so well last time", "timestamp": "10:06"},

    # Mixed tone
    {"speaker": "User2", "cleaned_message": "Thanks for replying so fast… really helpful", "timestamp": "10:07"},

    # Emotional shift
    {"speaker": "User1", "cleaned_message": "This is getting really frustrating now", "timestamp": "10:08"},

    # Edge case
    {"speaker": "User2", "cleaned_message": "", "timestamp": "10:09"},
]

# Run analyzer
result = analyzer.analyze(conversation)

# =========================
# PRINT MESSAGE-LEVEL OUTPUT
# =========================
print("\n===== MESSAGE ANALYSIS =====\n")

for msg in result["messages"]:
    print(f"[{msg['index']}] {msg['speaker']} ({msg['timestamp']})")
    print("Text:", msg["text"])
    print("Label:", msg["label"])
    print("Sarcastic:", msg["is_sarcastic"], "| Neutral:", msg["is_neutral"])
    print("Emotion Intensity:", msg["emotion_intensity"])
    print("-" * 50)

# =========================
# PRINT USER SUMMARY
# =========================
print("\n===== USER SUMMARY =====\n")

for user, data in result["users"].items():
    print(f"User: {user}")
    print("Message Count:", data["message_count"])
    print("Sarcasm Frequency:", round(data["sarcasm_frequency"], 2))
    print("Manipulation Frequency:", round(data["manipulation_frequency"], 2))
    print("Neutral Frequency:", round(data["neutral_frequency"], 2))
    print("Avg Emotion Intensity:", round(data["avg_emotion_intensity"], 2))
    print("Emotion Drift:", [round(x, 3) for x in data["emotion_drift"]])
    print("=" * 60)