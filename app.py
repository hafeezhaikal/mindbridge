from flask import Flask, render_template, request, jsonify, session
import math
import random
import json
from datetime import datetime
import subprocess
import threading
import time
import re

app = Flask(__name__)
app.secret_key = "mindbridge_secret_2024"

# ─── Sigmoid & Dynamic Perceptron ────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def compute_wellbeing_score(reaction_time_dev, nlp_sentiment, cv_facial, gameplay_error_rate, emotion="neutral"):
    """
    Dynamic weights based on detected emotion.
    Default (neutral): w1=0.2, w2=0.4, w3=0.3, w4=0.1, bias=-0.5
    Emotions override weights and bias.
    """
    # Default weights (neutral)
    w1, w2, w3, w4 = 0.2, 0.4, 0.3, 0.1
    bias = -0.5

    # Emotion‑specific weight vectors
    if emotion == "happy":
        w1, w2, w3, w4 = 0.05, 0.05, 0.8, 0.1   # CV dominates (80%)
        bias = -1.4                              # Strong negative bias for happy
    elif emotion == "sad":
        w1, w2, w3, w4 = 0.1, 0.6, 0.2, 0.1
        bias = -0.3
    elif emotion == "angry":
        w1, w2, w3, w4 = 0.1, 0.2, 0.6, 0.1
        bias = -0.2
    elif emotion == "fearful":
        w1, w2, w3, w4 = 0.1, 0.5, 0.3, 0.1
        bias = -0.3
    elif emotion == "surprised":
        w1, w2, w3, w4 = 0.1, 0.3, 0.5, 0.1
        bias = -0.4

    z = (w1 * reaction_time_dev) + (w2 * nlp_sentiment) + (w3 * cv_facial) + (w4 * gameplay_error_rate) + bias
    return round(sigmoid(z), 4), {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "b": bias}

# ─── NLP Sentiment Mapping ──────────────────────────────────────────────────

EMOTION_NLP_SCORES = {
    "happy":     0.90,
    "excited":   0.85,
    "calm":      0.75,
    "okay":      0.60,
    "neutral":   0.50,
    "tired":     0.35,
    "confused":  0.30,
    "bored":     0.28,
    "lonely":    0.22,
    "sad":       0.18,
    "anxious":   0.15,
    "angry":     0.12,
    "stressed":  0.10,
    "hopeless":  0.05,
    "overwhelmed": 0.08,
}

KEYWORD_INTENT_MAP = {
    "assignment": "Academic Burnout",
    "exam": "Academic Pressure",
    "study": "Academic Pressure",
    "fail": "Academic Fear",
    "university": "Academic Burnout",
    "school": "Academic Pressure",
    "tired": "Physical Fatigue",
    "sleep": "Sleep Deprivation",
    "lonely": "Social Isolation",
    "friend": "Social Concern",
    "family": "Family Stress",
    "money": "Financial Anxiety",
    "job": "Career Anxiety",
    "future": "Future Uncertainty",
    "hopeless": "Depressive Ideation",
    "useless": "Low Self-Worth",
    "angry": "Emotional Dysregulation",
    "panic": "Acute Anxiety",
    "worthless": "Low Self-Worth",
    "overwhelm": "Cognitive Overload",
}

def analyse_free_text(text):
    if not text:
        return None, 0.50
    text_lower = text.lower()
    
    negative_words = ["not", "don't", "can't", "won't", "never", "no"]
    has_negation = any(w in text_lower for w in negative_words)
    positive_words = ["happy", "good", "great", "fine", "okay", "well", "better", "smooth", "smoothly", "excited", "wonderful"]
    negative_sentiment_words = ["bad", "worse", "terrible", "awful", "horrible",
                                 "tired", "stressed", "anxious", "sad", "angry",
                                 "overwhelm", "hopeless", "fail", "lonely", "lost"]
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_sentiment_words if w in text_lower)
    if has_negation and pos_count > 0:
        neg_count += pos_count
        pos_count = 0
    total = pos_count + neg_count if (pos_count + neg_count) > 0 else 1
    raw_score = pos_count / total
    nlp_score = max(0.05, min(0.95, raw_score))
    
    if nlp_score > 0.7:
        return None, round(nlp_score, 2)
    
    detected_intent = None
    for keyword, intent in KEYWORD_INTENT_MAP.items():
        if keyword in text_lower:
            detected_intent = intent
            break
    return detected_intent, round(nlp_score, 2)

# ─── Game Selection Logic (Memory Match threshold lowered to 0.30) ──────────

def get_game_recommendation(score):
    """
    Maps wellbeing score (0-1) to an activity.
    - ≤0.30 → Creative Expression
    - 0.31–0.40 → Memory Match
    - 0.41–0.55 → Focus Tap
    - >0.55 → Breathing Exercise
    """
    if score > 0.55:
        return {
            "type": "breathing",
            "name": "Breathing Exercise",
            "label": "Calm and Relax",
            "description": "Guided box breathing to reduce cortisol and reset your nervous system.",
            "reason": "Your stress indicators are elevated. This breathing exercise activates the parasympathetic nervous system to lower anxiety."
        }
    elif score > 0.40:
        return {
            "type": "focus_tap",
            "name": "Focus Tap Challenge",
            "label": "Sharpen Your Focus",
            "description": "Tap the targets as they appear to rebuild your concentration and reaction time.",
            "reason": "Your response patterns suggest moderate stress. This game reactivates cognitive engagement circuits."
        }
    elif score > 0.20:
        return {
            "type": "memory",
            "name": "Memory Match Game",
            "label": "Train Your Memory",
            "description": "Match pairs of cards to stimulate working memory and gentle mental engagement.",
            "reason": "You're in a calm state. Memory activities provide low-stress mental stimulation."
        }
    else:
        return {
            "type": "creative",
            "name": "Creative Expression",
            "label": "Express Yourself",
            "description": "Answer expressive prompts to process your emotions through creative choices.",
            "reason": "You appear stable and happy. Creative expression reinforces positive emotional awareness."
        }

def get_status(score):
    if score > 0.75:
        return {"label": "High Stress", "level": "danger", "color": "#FF6B6B"}
    elif score > 0.60:
        return {"label": "Moderate Stress", "level": "warning", "color": "#FFB347"}
    elif score > 0.50:
        return {"label": "Mild Sadness", "level": "caution", "color": "#FFD700"}
    elif score > 0.35:
        return {"label": "Slightly Low", "level": "info", "color": "#87CEEB"}
    else:
        return {"label": "Stable / Happy", "level": "success", "color": "#7BE8A6"}

# ─── Chatbot Response Engine (unchanged) ────────────────────────────────────

CHATBOT_RESPONSES = {
    "high_stress": [
        "I hear you, and it is completely okay to feel that way. Your wellness dashboard shows elevated stress indicators. Let's work through this together.",
        "Thank you for being honest about how you feel. High stress is your mind and body asking for support. You reached out, and that matters.",
        "You are not alone in this. Many young people experience these feelings. I have prepared a breathing exercise designed specifically for moments like these.",
    ],
    "moderate_stress": [
        "I understand things feel heavy right now. Your check-in results show some signs of stress that we can address together.",
        "It sounds like you are carrying quite a bit. That is okay. Let us take a small step toward feeling better.",
        "Thank you for checking in. Your wellbeing matters. I have a focus activity that can help clear your mind a little.",
    ],
    "mild": [
        "Thank you for sharing. It sounds like things are a bit up and down today. That is completely normal.",
        "I can see from your check-in that you might benefit from a gentle mental boost. I have just the activity for that.",
    ],
    "stable": [
        "It is wonderful to hear you are doing well! Keeping up with regular check-ins helps you stay that way.",
        "You are in a great headspace. Let us do something fun to keep that positive energy going.",
    ],
    "coping_strategies": {
        "Academic Burnout": "Try breaking your study sessions into 25-minute focused blocks with 5-minute breaks. This is called the Pomodoro technique and it significantly reduces burnout.",
        "Academic Pressure": "Remember that your worth is not defined by your grades. Talk to a counsellor or trusted lecturer if the pressure feels unmanageable.",
        "Physical Fatigue": "Prioritise sleep above all else. Even a 20-minute nap can restore cognitive function significantly.",
        "Social Isolation": "Reaching out, even in small ways like a text to a friend, activates social bonding hormones that directly improve mood.",
        "Family Stress": "It can help to write down your feelings before a difficult conversation. It organises your thoughts and reduces emotional reactivity.",
        "Financial Anxiety": "Consider speaking to your institution's student welfare office. Many support resources exist that students are unaware of.",
        "Future Uncertainty": "Ground yourself in what you can control today. Write down three small actions you can take this week.",
        "Cognitive Overload": "Try a brain dump exercise: write every thought racing through your mind onto paper. Externalising thoughts reduces their mental weight.",
        "Low Self-Worth": "Your feelings are valid, but they are not permanent facts. Consider speaking with a counsellor who can support you properly.",
        "Acute Anxiety": "Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "Depressive Ideation": "I want you to know that support is available. Please consider reaching out to a mental health professional or a trusted adult.",
    },
    "general_followup": [
        "Is there anything specific on your mind that you would like to talk about?",
        "How long have you been feeling this way?",
        "Have you spoken to anyone about how you are feeling?",
        "What usually helps you feel better when you are going through a tough time?",
        "On a scale of 1 to 10, how would you rate your stress level right now?",
    ]
}

def get_chatbot_response(score, intent, user_message=""):
    status = get_status(score)
    if score > 0.75:
        key = "high_stress"
    elif score > 0.50:
        key = "moderate_stress"
    elif score > 0.35:
        key = "mild"
    else:
        key = "stable"
    intro = random.choice(CHATBOT_RESPONSES[key])
    coping = ""
    if intent and intent in CHATBOT_RESPONSES["coping_strategies"]:
        coping = CHATBOT_RESPONSES["coping_strategies"][intent]
    game = get_game_recommendation(score)
    return {
        "intro": intro,
        "coping": coping,
        "intent": intent,
        "game": game,
        "status": status,
        "score_pct": round(score * 100)
    }

# ─── Enhanced Mental Health Responses for Live Chat ─────────────────────────

MENTAL_HEALTH_RESPONSES = {
    "stress": "Stress is your body's natural response to pressure. Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. It interrupts the stress loop in seconds.",
    "anxiety": "Anxiety often comes from worrying about the future. A quick breathing exercise: inhale for 4 seconds, hold for 4, exhale for 6. This activates your vagus nerve and calms the nervous system. Would you like a guided exercise?",
    "panic": "If you're having a panic attack, focus on slow, deep breaths. Place one hand on your chest and one on your belly, and breathe into your belly. You are safe. This feeling will pass.",
    "sad": "Feeling sad is a human experience. Sometimes just naming it helps. What does sadness feel like in your body right now? (Heavy chest? Low energy?)",
    "depressed": "Depression can make everything feel heavy. Please know that this is not your fault and you deserve support. Reaching out to a therapist or counselor can be life-changing. Would you like me to share some resources?",
    "hopeless": "I hear that you're feeling hopeless. That's a very difficult place to be. Please consider contacting a crisis helpline – they are trained to support you. In the US: 988 (Suicide and Crisis Lifeline). UK: 111. International: find a local number at befriendly.org.",
    "sleep": "Sleep hygiene can transform your mood. Try: same bedtime each night, no screens 30 minutes before bed, a cool dark room, and a wind-down routine like reading or gentle stretching. Would you like a sleep meditation?",
    "tired": "Fatigue can be physical or emotional. If you've been pushing hard, a 20-minute rest (even lying quietly) resets your cognitive batteries. Be kind to yourself.",
    "lonely": "Loneliness is a signal that you need connection. Reaching out – even a simple text to a friend – can ease it. You can also try joining a club, volunteering, or a peer support group. You are not alone.",
    "friend": "Friendships take effort. Start small: send a meme, ask how their day was, or suggest a video call. Connection is a skill, not a talent.",
    "exam": "Exam stress is common. Break your study into 25-minute blocks (Pomodoro). After each block, take a 5-minute break – walk, stretch, drink water. Reward yourself after each cycle.",
    "assignment": "Overwhelmed by assignments? Write down every task, then pick the smallest one and do it for just 10 minutes. Starting is the hardest part.",
    "burnout": "Burnout is emotional exhaustion from prolonged stress. Recovery requires rest, boundaries, and often professional support. It's okay to step back.",
    "worthless": "Your worth is not determined by your productivity or others' opinions. You matter simply because you exist. If these thoughts are persistent, please talk to a professional.",
    "useless": "That's a painful thought. Sometimes our inner critic is loud. Try writing down one small thing you did today that was kind or helpful – even just getting up counts.",
    "angry": "Anger is a signal that something matters to you. Instead of suppressing it, try taking a few deep breaths and then using 'I feel' statements. Would you like a quick anger release exercise?",
    "rage": "When anger feels overwhelming, remove yourself from the situation temporarily. Splash cold water on your face, go outside, or scream into a pillow. You are allowed to feel angry, but you also deserve to feel calm.",
    "family": "Family relationships can be complex. If conversations are hard, try writing a letter first. Set boundaries kindly: 'I love you, but I need to take a break right now.'",
    "breakup": "Heartbreak is real grief. Allow yourself to feel it, but also reach out to friends. Avoid isolation. This pain will ease with time, though it doesn't feel that way now.",
    "cope": "Healthy coping includes: exercise, sleep, talking to someone, creative outlets (art, music, writing), nature, and mindfulness. What activities have helped you in the past?",
    "breathe": "Here's a simple breathing exercise: Inhale 4 seconds, hold 7 seconds, exhale 8 seconds. Repeat 5 times. This lowers cortisol immediately.",
    "grounding": "Use the 5-4-3-2-1 technique: 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Do it now – it pulls your brain into the present.",
    "better": "I'm so glad you're feeling better! That's wonderful. Remember what helped you get here – you can use those tools anytime.",
    "good": "That's great to hear. Keeping a regular check‑in like this builds emotional resilience. Well done.",
    "how it works": "I use a simple AI model: I look at your check‑in (emotion + text), your face scan expression, and your reaction times. Each is weighted differently, and the final score is run through a sigmoid function to give a stress probability between 0 and 1. Then I match your words to mental health topics.",
    "ai": "I'm not a human, but I'm trained to recognise feelings and offer evidence‑based suggestions. I don't replace a therapist, but I can be a first step.",
    "help": "You deserve support. If you're in crisis, please contact a helpline: (US) 988 – Suicide & Crisis Lifeline. (UK) 111 – mental health triage. (Global) find a local number at findahelpline.com. You matter.",
    "suicide": "I'm really glad you reached out. That takes courage. Please contact a crisis line immediately: 988 (US) or your local emergency number. You are not alone, and there is help.",
    "game": "Based on your current wellbeing score, I recommend the activity shown below. It's designed to gently improve your mood or focus.",
    "play": "Activities are most helpful when you're open to them. Even 5 minutes of a game can shift your brain state.",
}

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

@app.route("/checkin")
def checkin():
    return render_template("checkin.html")

@app.route("/api/analyse-checkin", methods=["POST"])
def analyse_checkin():
    data = request.get_json()
    emotion = data.get("emotion", "neutral").lower()
    free_text = data.get("free_text", "")
    reaction_time_ms = data.get("reaction_time_ms", 500)

    intent, text_nlp_score = analyse_free_text(free_text)
    emotion_nlp_score = EMOTION_NLP_SCORES.get(emotion, 0.50)
    if free_text.strip():
        final_nlp = (text_nlp_score * 0.6 + emotion_nlp_score * 0.4)
    else:
        final_nlp = emotion_nlp_score

    baseline_ms = 500
    deviation = (baseline_ms - reaction_time_ms) / 1000.0
    deviation = max(-1.0, min(1.0, deviation))

    cv_score = data.get("cv_score", round(random.uniform(0.3, 0.8), 2))
    error_rate = data.get("error_rate", 0)
    detected_emotion = data.get("detected_emotion", "neutral").lower()

    score, dynamic_weights = compute_wellbeing_score(deviation, final_nlp, cv_score, error_rate, detected_emotion)
    status = get_status(score)
    game = get_game_recommendation(score)
    chatbot = get_chatbot_response(score, intent)

    result = {
        "score": score,
        "score_pct": round(score * 100),
        "status": status,
        "game": game,
        "nlp": {
            "emotion_score": round(emotion_nlp_score, 2),
            "text_score": round(text_nlp_score, 2),
            "final_score": round(final_nlp, 2),
            "intent": intent
        },
        "cv_score": cv_score,
        "reaction_deviation": round(deviation, 3),
        "error_rate": error_rate,
        "chatbot": chatbot,
        "detected_emotion": detected_emotion,
        "formula": {
            "z": round((dynamic_weights["w1"] * deviation) + (dynamic_weights["w2"] * final_nlp) + (dynamic_weights["w3"] * cv_score) + (dynamic_weights["w4"] * error_rate) + dynamic_weights["b"], 4),
            "weights": dynamic_weights
        }
    }

    session["analysis"] = result
    return jsonify(result)

@app.route("/scan")
def scan():
    return render_template("scan.html")

@app.route("/analysis")
def analysis():
    data = session.get("analysis", {})
    return render_template("analysis.html", data=json.dumps(data))

@app.route("/chatbot")
def chatbot():
    data = session.get("analysis", {})
    return render_template("chatbot.html", data=json.dumps(data))

@app.route("/api/chatbot-reply", methods=["POST"])
def chatbot_reply():
    payload = request.get_json()
    user_msg = payload.get("message", "").lower()
    wellbeing_score = payload.get("score", 0.5)
    intent = payload.get("intent", None)

    # Crisis detection (highest priority)
    crisis_keywords = ["suicide", "kill myself", "want to die", "end my life", "can't go on", "helpless"]
    if any(kw in user_msg for kw in crisis_keywords):
        reply = ("I hear that you're in a lot of pain. Please, reach out to a crisis line right now. "
                 "In the US: 988 (Suicide and Crisis Lifeline). UK: 111. International: findahelpline.com. "
                 "You are not alone – people care about you. 💙")
        return jsonify({"reply": reply})

    # --- Game request detection (added) ---
    game_keywords = ["game", "play", "activity", "recommend a game", "suggest a game"]
    if any(kw in user_msg for kw in game_keywords):
        game = get_game_recommendation(wellbeing_score)
        reply = f"I recommend the **{game['name']}**. {game['description']}\n\nTap the button below to start."
        return jsonify({"reply": reply, "game": game})

    for keyword, response in MENTAL_HEALTH_RESPONSES.items():
        if keyword in user_msg:
            if keyword == "game":
                game = get_game_recommendation(wellbeing_score)
                reply = f"{response}\n\nI recommend the **{game['name']}**. {game['description']}"
            else:
                reply = response
            return jsonify({"reply": reply})

    if intent and intent in CHATBOT_RESPONSES.get("coping_strategies", {}):
        reply = CHATBOT_RESPONSES["coping_strategies"][intent]
        return jsonify({"reply": reply})

    fallbacks = [
        "Thank you for sharing that. How has this been affecting your daily life?",
        "I appreciate you opening up. Is there a specific area of your life where you feel this most strongly?",
        "That is important to acknowledge. What kind of support feels most helpful to you right now?",
        "I am listening. Tell me more about what has been going on.",
        "You are doing well by talking about this. What would feel like a small win for you today?",
    ]
    reply = random.choice(fallbacks)
    return jsonify({"reply": reply})

@app.route("/game/<game_type>")
def game(game_type):
    data = session.get("analysis", {})
    valid = ["breathing", "focus_tap", "memory", "creative"]
    if game_type not in valid:
        game_type = "breathing"
    return render_template(f"game_{game_type}.html", data=json.dumps(data))

@app.route("/api/save-game-result", methods=["POST"])
def save_game_result():
    payload = request.get_json()
    session["game_result"] = payload
    return jsonify({"ok": True})

@app.route("/result")
def result():
    analysis = session.get("analysis", {})
    game_result = session.get("game_result", {})
    return render_template("result.html",
                           data=json.dumps(analysis),
                           game_data=json.dumps(game_result))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)