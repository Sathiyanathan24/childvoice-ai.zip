# 🎙️ ChildVoice AI

AI-driven speech and language development screening for children, built with **Streamlit** and custom **CSS**.

> **Disclaimer:** ChildVoice AI is an AI-assisted screening tool, not a medical diagnosis.
> Professional evaluation should be sought for clinical decisions.

## Features

- Guided assessment flow: child details → assessment type → record/upload → AI analysis
- In-browser recording (`st.audio_input`) or audio upload (wav/mp3/m4a, ≤ 10 MB)
- Results page with overall score, category scores and speech metrics
- Detailed, downloadable report
- Parent guide, professional dashboard, resources, contact and login/register pages

## Project structure

```
childvoice-ai/
├── app.py                 # Streamlit app (pages, state, analysis hook)
├── styles/style.css       # All custom CSS
├── models/                # Trained model files (git-ignored)
├── tests/                 # Unit tests
├── .streamlit/config.toml # Theme and server settings
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/childvoice-ai.git
cd childvoice-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Demo login: `demo@childvoice.ai` / `demo123`

## Connecting your real model

`analyse_audio()` in `app.py` currently returns **simulated** scores so the full flow works.
To go live:

1. Extract features (MFCC, pitch, formants) with Librosa.
2. Load your TensorFlow CNN + BiGRU model from `models/`.
3. Return the same dictionary shape (`overall`, `level`, `cats`, `metrics`, `date`).

## Roadmap

- [ ] Real model inference
- [ ] MySQL storage with hashed passwords
- [ ] PDF report export
- [ ] Multi-language prompts

## License

MIT


Live Demo: https://childvoice-aizip-ixztey8iuqsecqdhbsddaa.streamlit.app/
