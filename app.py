from flask import Flask, request, render_template
import joblib
import re
from transformers import BartTokenizer, BartForConditionalGeneration

specialty_clf = joblib.load("specialty_classifier.pkl")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

app = Flask(__name__)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def bart_summary(text):
    inputs = bart_tokenizer([text], return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    cleaned = preprocess_text(user_input)
    summary = bart_summary(cleaned)
    specialty = specialty_clf.predict([cleaned])[0]
    return render_template("index.html", user_input=user_input, summary=summary, specialty=specialty)

if __name__ == "__main__":
    app.run(debug=True)
