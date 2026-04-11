import os

os.environ.pop("MPLBACKEND", None)

import matplotlib

matplotlib.use("TkAgg")  # or "Qt5Agg"
print("Backend:", matplotlib.get_backend())

from pipeline.analyzer import ConversationAnalyzer
from pipeline.insights import analyze_insights
from pipeline.visualizer import (
    build_visualization_data,
    run_all_plots,
    plot_plutchik_wheel
)

# Initialize analyzer
analyzer = ConversationAnalyzer()

# Complex 4-user conversation (final stress test)
conversation = [
    {"speaker": "Alice", "cleaned_message": "Hey everyone, just wanted to sync on the progress", "timestamp": "11:00"},
    {"speaker": "Bob", "cleaned_message": "Yeah sure", "timestamp": "11:01"},
    {"speaker": "Charlie", "cleaned_message": "Morning", "timestamp": "11:02"},
    {"speaker": "Diana", "cleaned_message": "Let’s keep this quick", "timestamp": "11:03"},

    {"speaker": "Charlie", "cleaned_message": "Okay noted", "timestamp": "11:04"},
    {"speaker": "Diana", "cleaned_message": "Go ahead", "timestamp": "11:05"},

    {"speaker": "Bob", "cleaned_message": "Oh nice, another update meeting. Love those", "timestamp": "11:06"},
    {"speaker": "Alice", "cleaned_message": "It won’t take long", "timestamp": "11:07"},

    {"speaker": "Bob", "cleaned_message": "Yeah because these always finish on time", "timestamp": "11:08"},
    {"speaker": "Charlie", "cleaned_message": "Let’s just go through the points", "timestamp": "11:09"},

    {"speaker": "Diana", "cleaned_message": "If we had better coordination earlier, this wouldn’t be needed", "timestamp": "11:10"},
    {"speaker": "Alice", "cleaned_message": "That’s a bit unfair honestly", "timestamp": "11:11"},

    {"speaker": "Diana", "cleaned_message": "I mean I’m just saying what everyone is thinking", "timestamp": "11:12"},
    {"speaker": "Bob", "cleaned_message": "Yeah totally, super helpful as always", "timestamp": "11:13"},

    {"speaker": "Charlie", "cleaned_message": "Let’s focus on solutions instead", "timestamp": "11:14"},
    {"speaker": "Alice", "cleaned_message": "This is getting frustrating now", "timestamp": "11:15"},

    {"speaker": "Diana", "cleaned_message": "I just want us to do better, that’s all", "timestamp": "11:16"},
    {"speaker": "Bob", "cleaned_message": "Perfect, everything is going great clearly", "timestamp": "11:17"},

    {"speaker": "Charlie", "cleaned_message": "Alright, action items please", "timestamp": "11:18"},
    {"speaker": "Alice", "cleaned_message": "Okay let’s just finish this properly", "timestamp": "11:19"},
]

# Step 1: Analyze
analysis = analyzer.analyze(conversation)

# Step 2: Generate insights
insights = analyze_insights(analysis)

# Step 3: Build visualization payload
viz_data = build_visualization_data(analysis, insights)

# Step 4: Print summaries (sanity check)
print("\n===== FINAL SYSTEM OUTPUT =====\n")
for user, data in insights.items():
    print(f"{user}:")
    print("Summary:", data["summary"])
    print("Tone:", data["emotional_tone"])
    print("Sarcasm:", data["sarcasm_level"])
    print("Manipulation:", data["manipulation_level"])
    print("Neutral:", data["neutral_level"])
    print("-" * 50)

# Step 5: Run all plots
print("\nLaunching visualizations...\n")
run_all_plots(viz_data)

# Step 6: Show Plutchik wheels separately (optional clarity)
print("\nGenerating Plutchik emotion wheels...\n")
users = analysis.get("users", {})
for speaker, user_data in users.items():
    emotion_avg = user_data.get("emotion_avg", [])
    plot_plutchik_wheel(emotion_avg, title=f"{speaker} Emotion Profile")