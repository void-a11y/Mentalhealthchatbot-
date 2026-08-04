# 🧠 Mental Health AI Chatbot

An AI-powered mental health chatbot designed to provide supportive and empathetic conversations. The chatbot uses Large Language Models (LLMs) to interact with users, detect crisis situations, retrieve relevant mental health resources, and maintain conversation logs for analysis.

> **Disclaimer:** This chatbot is intended for educational and research purposes only. It is **not** a substitute for professional mental health care or emergency services.

---

## Features

- 💬 AI-powered conversational chatbot
- 🚨 Crisis keyword detection
- 📄 Document/resource retrieval
- 📝 Conversation logging
- 🤖 Modular chatbot architecture
- 🔒 Environment variable support for API keys

---

## Project Structure

```
Mentalhealthchatbot/
│
├── main.py              # Application entry point
├── chatengine.py        # Chatbot conversation engine
├── crisis.py            # Crisis detection module
├── doc_engine.py        # Resource/document retrieval
├── logger.py            # Conversation logging
├── models.py            # AI model configuration
├── requirements.txt     # Project dependencies
└── README.md
```

---

## Tech Stack

- Python
- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- CSV Logging
- Modular Python Architecture

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/void-a11y/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file and add your API keys.

Example:

```env
API_KEY=your_api_key_here
```

### 4. Run the chatbot

```bash
python main.py
```

---

## Workflow

```
User
   │
   ▼
Chat Engine
   │
   ├── Crisis Detection
   │
   ├── Document Retrieval
   │
   └── AI Model
        │
        ▼
 Response
        │
        ▼
 Conversation Logger
```

---

## Future Improvements

- Voice-based interaction
- Emotion detection
- Multi-language support
- User authentication
- Therapist dashboard
- Database integration
- Conversation analytics

---

## Author

**Suwarnika Srivastava**

B.Tech Computer Science & Engineering (Cloud Computing)

---

## License

This project is licensed under the MIT License.
