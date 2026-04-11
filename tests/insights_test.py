from pipeline.analyzer import ConversationAnalyzer
from pipeline.insights import analyze_insights

analyzer = ConversationAnalyzer()

conversation = [
    # Initial neutral setup
    {"speaker": "Alice", "cleaned_message": "Hey everyone, just wanted to sync on the progress", "timestamp": "11:00"},
    {"speaker": "Bob", "cleaned_message": "Yeah sure", "timestamp": "11:01"},
    {"speaker": "Charlie", "cleaned_message": "Morning", "timestamp": "11:02"},
    {"speaker": "Diana", "cleaned_message": "Let’s keep this quick", "timestamp": "11:03"},

    # Neutral / filler
    {"speaker": "Charlie", "cleaned_message": "Okay noted", "timestamp": "11:04"},
    {"speaker": "Diana", "cleaned_message": "Go ahead", "timestamp": "11:05"},

    # Subtle sarcasm (Bob starts)
    {"speaker": "Bob", "cleaned_message": "Oh nice, another update meeting. Love those", "timestamp": "11:06"},
    {"speaker": "Alice", "cleaned_message": "It won’t take long", "timestamp": "11:07"},

    # Passive sarcasm escalation
    {"speaker": "Bob", "cleaned_message": "Yeah because these always finish on time", "timestamp": "11:08"},

    # Charlie stays neutral/stabilizing
    {"speaker": "Charlie", "cleaned_message": "Let’s just go through the points", "timestamp": "11:09"},

    # Diana = subtle manipulator (hard case)
    {"speaker": "Diana", "cleaned_message": "If we had better coordination earlier, this wouldn’t be needed", "timestamp": "11:10"},

    # Alice reacts emotionally
    {"speaker": "Alice", "cleaned_message": "That’s a bit unfair honestly", "timestamp": "11:11"},

    # Diana doubles down (manipulation)
    {"speaker": "Diana", "cleaned_message": "I mean I’m just saying what everyone is thinking", "timestamp": "11:12"},

    # Bob sarcastic again
    {"speaker": "Bob", "cleaned_message": "Yeah totally, super helpful as always", "timestamp": "11:13"},

    # Charlie stabilizing again
    {"speaker": "Charlie", "cleaned_message": "Let’s focus on solutions instead", "timestamp": "11:14"},

    # Alice emotional drift
    {"speaker": "Alice", "cleaned_message": "This is getting frustrating now", "timestamp": "11:15"},

    # Diana disguised manipulation
    {"speaker": "Diana", "cleaned_message": "I just want us to do better, that’s all", "timestamp": "11:16"},

    # Bob sarcasm + negativity
    {"speaker": "Bob", "cleaned_message": "Perfect, everything is going great clearly", "timestamp": "11:17"},

    # Charlie neutral leadership
    {"speaker": "Charlie", "cleaned_message": "Alright, action items please", "timestamp": "11:18"},

    # Alice calming down
    {"speaker": "Alice", "cleaned_message": "Okay let’s just finish this properly", "timestamp": "11:19"},
]

analysis = analyzer.analyze(conversation)
insights = analyze_insights(analysis)

print("\n===== 4-PERSON COMPLEX INSIGHTS =====\n")

for user, data in insights.items():
    print(f"{user}:")
    print("Summary:", data["summary"])
    print("Dominant Emotion:", data["dominant_emotion"])
    print("Tone:", data["emotional_tone"])
    print("Stability:", data["emotional_stability"])
    print("Sarcasm Level:", data["sarcasm_level"])
    print("Manipulation Level:", data["manipulation_level"])
    print("Neutral Level:", data["neutral_level"])
    print("Intensity:", data["emotional_intensity_level"])
    print("Risk Flags:", data["risk_flags"])
    print("-" * 60)