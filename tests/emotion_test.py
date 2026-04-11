from models.emotion_model import EmotionModel 

model = EmotionModel()

print(model.predict("I love you so much"))
print(model.predict("Yeah sure, whatever 🙄"))
print(model.predict("I'm scared this will fail"))