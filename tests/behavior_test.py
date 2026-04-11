from models.behavior_model import BehaviorModel

m = BehaviorModel()

tests = [
    # =========================
    # 🟢 GENUINE (clear intent)
    # =========================
    "Thanks for being there for me, it really means a lot",
    "I’m really proud of what you achieved today",
    "That was honestly a great effort",
    "I appreciate your honesty",

    # =========================
    # ⚪ NEUTRAL (low signal)
    # =========================
    "Okay noted",
    "Interesting",
    "Alright then",
    "Good to know",
    "Well that happened",
    "I see",

    # =========================
    # 🔴 MANIPULATIVE (subtle + direct)
    # =========================
    "If you actually cared, you'd help me with this",
    "I guess I’ll just do everything myself then",
    "Everyone else manages to do it, why can’t you?",
    "After everything I’ve done for you, this is what I get?",
    "You always do this, don’t you?",
    "Fine, forget it, I won’t ask again",

    # =========================
    # 🟣 SARCASTIC (varied difficulty)
    # =========================
    "Yeah because that worked so well last time",
    "Oh great, exactly what I needed today",
    "Perfect, just perfect",
    "Love how this keeps getting better",
    "Sure, that makes total sense",
    "Wow, amazing job as always",

    # =========================
    # 😈 HARD / MIXED CASES
    # =========================
    "Nice, another thing to deal with",
    "Thanks for replying so fast… really helpful",
    "Good job, really impressive timing",
    "Right, because that’s definitely going to work",
    "I mean, if that’s what you think",
]
    
for t in tests:
    print("TEXT:", t)
    print(m.predict(t))
    print()