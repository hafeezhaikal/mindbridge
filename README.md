# MindBridge — AI-Powered Youth Wellbeing System

Gamified mental health support using NLP, Computer Vision simulation, adaptive mini-games, and a conversational AI chatbot.

---

## Setup & Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask server
```bash
python app.py
```

### 3. Open in your browser
```
http://localhost:5000
```

---

## System Flow

```
Welcome → Check-in (NLP + Reaction Time) → Face Scan (CV) → AI Analysis → Chatbot → Game → Report
```

---

## AI Decision Pipeline

The system uses a Perceptron model:

```
Z = (0.2 * x1) + (0.4 * x2) + (0.3 * x3) + (0.1 * x4) - 0.5
Score = sigmoid(Z) = 1 / (1 + e^-Z)
```

| Input | Description                  | Weight |
|-------|------------------------------|--------|
| x1    | Reaction time deviation      | 0.2    |
| x2    | NLP sentiment score          | 0.4    |
| x3    | CV facial expression score   | 0.3    |
| x4    | Gameplay error rate          | 0.1    |
| b     | Bias (decision boundary)     | -0.5   |

---

## Rule-Based Game Deployment

| Score Range | Status           | Game Deployed         |
|-------------|------------------|-----------------------|
| > 0.75      | High Stress      | Breathing Exercise    |
| 0.51 - 0.75 | Mild Sadness     | Focus Tap Challenge   |
| 0.31 - 0.50 | Slightly Low     | Memory Match Game     |
| <= 0.30     | Stable / Happy   | Creative Expression   |

---

## Features

- **12 Emotion Options**: Happy, Excited, Calm, Okay, Tired, Sad, Anxious, Stressed, Angry, Lonely, Confused, Hopeless
- **NLP Intent Detection**: 15 keyword-mapped intents (Academic Burnout, Social Isolation, etc.)
- **CV Face Scan**: Simulated facial landmark pipeline with 6-stage analysis
- **Perceptron Analysis**: Real formula with XAI breakdown
- **AI Chatbot**: Keyword-aware responses + coping strategies per intent
- **4 Mini Games**: Breathing, Focus Tap, Memory Match, Creative Expression
- **XAI Report**: Plain-language explanation of why each game was chosen
- **Edge AI Privacy**: Facial data stays on device (no upload)

---

## Project Structure

```
mindbridge/
  app.py                  # Flask routes + AI logic
  requirements.txt
  templates/
    base.html             # Shared layout + design system
    index.html            # Welcome page
    checkin.html          # Emotion + NLP + reaction time
    scan.html             # CV face scan simulation
    analysis.html         # Perceptron results + formula
    chatbot.html          # AI conversational companion
    game_breathing.html   # Breathing exercise (4x4 box)
    game_focus_tap.html   # Tap targets game
    game_memory.html      # Memory card matching
    game_creative.html    # Creative expression prompts
    result.html           # Full XAI report
```
