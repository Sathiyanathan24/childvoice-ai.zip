"""
ChildVoice AI – AI-driven speech & language development screening (Streamlit + custom CSS)

Run:  pip install -r requirements.txt
      streamlit run app.py

NOTE: analyse_audio() produces *simulated* scores so the full flow works end to end.
Replace it with your real TensorFlow (CNN + BiGRU) inference to go live.
"""
import io
import time
import wave
import zlib
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ChildVoice AI", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Design tokens + CSS
# ----------------------------------------------------------------------------
from pathlib import Path

import os

try:
    css_path = Path(__file__).parent / "styles" / "style.css"
    with open(css_path, encoding='utf-8') as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    print(f"CSS file not found at {css_path}")


def H(html: str):
    """Render HTML (strips indentation so markdown doesn't treat it as a code block)."""
    st.markdown("".join(l.strip() for l in html.splitlines()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# State & navigation
# ----------------------------------------------------------------------------
DEFAULTS = dict(page="Home", step=1, child={}, a_type="Overall Development", audio=None, audio_name=None,
                results=None, users={"demo@childvoice.ai": "demo123"}, user=None)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

NAV = [("Home", "Home"), ("About", "About"), ("Assessment", "Assessment"), ("For Parents", "Parents"),
       ("For Professionals", "Professionals"), ("Resources", "Resources"), ("Contact", "Contact"),
       ("Technology", "Technology")]


def go(page: str, step: int | None = None):
    st.session_state.page = page
    if step is not None:
        st.session_state.step = step


def nav():
    cols = st.columns([1.7, .7, .7, 1, 1.05, 1.35, .85, .85, 1, .8])
    with cols[0]:
        H('<div class="brand"><div class="logo">🎙️</div><div><b>ChildVoice AI</b>'
          '<small>Listen • Understand • Empower</small></div></div>')
    for i, (label, page) in enumerate(NAV):
        cols[i + 1].button(label, key=f"nav_{page}", on_click=go, args=(page,), use_container_width=True)
    if st.session_state.user:
        cols[9].button("Log out", key="nav_logout", type="primary", use_container_width=True,
                       on_click=lambda: st.session_state.update(user=None))
    else:
        cols[9].button("Login", key="nav_login_btn", type="primary", use_container_width=True, on_click=go, args=("Login",))
    # highlight active link
    active = st.session_state.page
    st.markdown(f"<style>.st-key-nav_{active} button{{color:#1E6BFF !important; "
                f"border-bottom:3px solid #1E6BFF !important;}}</style>", unsafe_allow_html=True)
    H('<div class="navline"></div>')


def feature_cards(items):
    cols = st.columns(len(items))
    for c, (ico, cls, title, text) in zip(cols, items):
        with c:
            H(f'<div class="card"><div class="icon {cls}">{ico}</div><h4>{title}</h4><p>{text}</p></div>')


def stepper(active: int):
    labels = ["Child Details", "Assessment Type", "Record / Upload", "Analyze"]
    html = '<div class="stepper">'
    for i, lab in enumerate(labels, 1):
        cls = "done" if i < active else "active" if i == active else ""
        html += f'<div class="step {cls}"><div class="dot">{"✓" if i < active else i}</div>{lab}</div>'
    H(html + "</div>")


def footer():
    H('''<div class="cta-band"><h2>Every Child's Voice Can Create a Brighter Tomorrow</h2>
        <p style="color:#C9D8F5">Early detection. Better support. Brighter futures.</p></div>''')
    H('''<div class="footer"><div><b>ChildVoice AI</b>Listen • Understand • Empower<br><br>
        Small Voices. Big Possibilities. 💙</div>
        <div><b>Quick Links</b><a>Home</a><a>About</a><a>Assessment</a><a>Resources</a></div>
        <div><b>Support</b><a>For Parents</a><a>For Professionals</a><a>Contact</a></div>
        <div><b>Legal</b><a>Privacy Policy</a><a>Terms of Service</a><a>Disclaimer</a></div></div>
        <div class="copy">© 2026 ChildVoice AI. All rights reserved.</div>''')


# ----------------------------------------------------------------------------
# Analysis (simulated – swap with your TensorFlow model)
# ----------------------------------------------------------------------------
def wav_duration(data: bytes):
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def analyse_audio(data: bytes, child: dict) -> dict:
    """Return dummy-but-stable scores derived from the audio bytes.
    TODO: extract MFCC/pitch/formants and run your CNN + BiGRU model here."""
    rng = np.random.default_rng(zlib.crc32(data) & 0xFFFFFFFF)
    cats = {k: int(rng.integers(62, 93)) for k in ["Pronunciation", "Fluency", "Speech Clarity", "Vocabulary", "Language"]}
    overall = int(round(np.mean(list(cats.values()))))
    dur = wav_duration(data) or float(rng.integers(20, 55))
    level = ("Typical Range", "green") if overall >= 75 else \
            ("Needs Observation", "orange") if overall >= 65 else ("Consider Professional Evaluation", "red")
    return dict(overall=overall, level=level, cats=cats, date=date.today().strftime("%d %b %Y"),
                metrics={"Speech Duration": f"{dur:.0f} sec", "Words Detected": int(dur * rng.uniform(.6, 1.1)),
                         "Speech Rate": f"{int(rng.integers(70, 110))} WPM",
                         "Pause Frequency": ["Low", "Normal", "Normal", "High"][int(rng.integers(0, 4))],
                         "Pronunciation Score": f"{cats['Pronunciation']}%"})


def bar_color(v):
    return "#22A06B" if v >= 78 else "#F59E0B" if v >= 68 else "#E5484D"


def report_html(child, res) -> bytes:
    rows = "".join(f"<tr><td>{k}</td><td>{v}%</td></tr>" for k, v in res["cats"].items())
    return f"""<html><head><meta charset='utf-8'><title>ChildVoice AI Report</title>
    <style>body{{font-family:Arial;max-width:720px;margin:2rem auto;color:#1B2B4A}}h1{{color:#0B2A5B}}
    td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}table{{border-collapse:collapse;width:100%}}
    .n{{background:#EAF2FF;padding:10px;border-radius:8px}}</style></head><body>
    <h1>ChildVoice AI – Assessment Report</h1>
    <p><b>Child:</b> {child.get('name','-')} &nbsp; <b>Age:</b> {child.get('age','-')} &nbsp;
    <b>Language:</b> {child.get('lang','-')}<br><b>Date:</b> {res['date']} &nbsp;
    <b>Type:</b> {st.session_state.a_type}<br><b>Overall:</b> {res['overall']}% – {res['level'][0]}</p>
    <h3>Category scores</h3><table><tr><th>Category</th><th>Score</th></tr>{rows}</table>
    <h3>Recommendations</h3><ul><li>Encourage regular speaking and storytelling.</li>
    <li>Use reading and conversation activities.</li><li>Monitor fluency and vocabulary development.</li>
    <li>Consider professional evaluation if concerns persist.</li></ul>
    <p class='n'>This is an AI-assisted screening report, not a medical diagnosis.</p></body></html>""".encode()


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    bars = "".join(f'<i style="height:{h}px"></i>' for h in [14, 30, 20, 40, 26, 36, 18, 28])
    H(f'''<div class="hero"><div>
        <h1>AI-Driven Speech and Language Development Assessment in Children</h1>
        <div class="tag">Early detection. Better support. Brighter futures.</div>
        <p>Using advanced AI and deep learning, we analyze children's speech and language development —
        making it easier, faster and more accessible for families and professionals.</p></div>
        <div class="hero-art"><div class="bubble">Every child's voice matters 💙</div>👦
        <div class="wave">{bars}</div><div class="bot">🤖</div></div></div>''')
    c1, c2, _ = st.columns([1.1, 1, 4])
    c1.button("Start Assessment", type="primary", on_click=go, args=("Assessment", 1), key="h_start")
    c2.button("Learn More", on_click=go, args=("About",), key="h_learn")
    H('<div style="height:.6rem"></div>')
    cols = st.columns(5)
    for c, (ico, cls, t) in zip(cols, [("🎤", "i-orange", "Speech Recording"), ("🧠", "i-green", "AI Analysis"),
                                       ("📊", "i-pink", "Development Assessment"), ("📄", "i-blue", "Detailed Report"),
                                       ("📋", "i-green", "Parent-friendly Results")]):
        c.markdown(f'<div class="feature"><div class="icon {cls}">{ico}</div><b>{t}</b></div>', unsafe_allow_html=True)
    H('<div class="pillrow"><span>🎧 Listen</span><span>🧩 Understand</span><span>🔍 Assess</span><span>💪 Empower</span></div>')


def page_about():
    H('''<div class="hero" style="grid-template-columns:1.4fr 1fr"><div><h1>About ChildVoice AI</h1>
        <div class="tag">Technology for a brighter tomorrow</div>
        <p>ChildVoice AI is a web-based system designed to analyze children's speech and language development
        using Artificial Intelligence and deep learning.</p></div>
        <div class="hero-art" style="height:200px"><div class="bubble">Better Communication,<br>A Brighter Tomorrow</div>👧</div></div>''')
    H('<div style="height:1rem"></div>')
    feature_cards([("👂", "i-pink", "Our Vision", "Every child's voice is heard."),
                   ("🌐", "i-teal", "Our Mission", "To make early detection accessible to all."),
                   ("👪", "i-purple", "Our Impact", "Supporting children, families and professionals.")])
    H('''<div class="disclaimer"><b>⚠️ Important Disclaimer</b><br>This system is an AI-assisted screening tool,
        not a medical diagnosis. Professional evaluation should be sought for clinical decisions.</div>''')


def page_assessment():
    s = st.session_state.step
    if s == 1:
        stepper(1)
        left, right = st.columns([1.5, 1])
        with left:
            st.markdown("### Child Information")
            st.caption("Please enter the details below to start the assessment.")
            with st.form("child_form"):
                c = st.session_state.child
                a, b = st.columns(2)
                name = a.text_input("Child Name *", c.get("name", ""), placeholder="Enter child's name")
                age = b.selectbox("Age *", [None] + list(range(2, 13)), format_func=lambda x: "Select age" if x is None else f"{x} years",
                                  index=([None] + list(range(2, 13))).index(c.get("age")) if c.get("age") else 0)
                a, b = st.columns(2)
                gender = a.radio("Gender (Optional)", ["Male", "Female", "Other"], horizontal=True, index=None)
                lang = b.selectbox("Primary Language *", ["Select language", "English", "Tamil", "Hindi", "Malayalam", "Telugu", "Kannada"])
                parent = st.text_input("Parent/Guardian Name *", c.get("parent", ""), placeholder="Enter parent/guardian name")
                if st.form_submit_button("Continue", type="primary"):
                    if not name.strip() or age is None or lang == "Select language" or not parent.strip():
                        st.error("Please fill in the child's name, age, primary language and guardian name.")
                    else:
                        st.session_state.child = dict(name=name.strip(), age=age, gender=gender, lang=lang, parent=parent.strip())
                        go("Assessment", 2)
                        st.rerun()
        with right:
            H('<div class="hero-art" style="height:340px;font-size:8rem"><div class="bubble">Small steps,<br>Big possibilities 💙</div>👦</div>')

    elif s == 2:
        stepper(2)
        st.markdown("### Select Assessment Type")
        types = [("🎤", "i-blue", "Speech", "Assess speech clarity, pronunciation and sound production."),
                 ("📖", "i-green", "Language", "Assess vocabulary, sentence structure and comprehension."),
                 ("🗣️", "i-orange", "Pronunciation", "Analyze specific speech sound production."),
                 ("🎼", "i-purple", "Fluency", "Assess speech rate, pauses and smoothness."),
                 ("📈", "i-teal", "Overall Development", "Comprehensive assessment of speech and language.")]
        cols = st.columns(5)
        for c, (ico, cls, t, d) in zip(cols, types):
            with c:
                H(f'<div class="card"><div class="icon {cls}">{ico}</div><h4>{t}</h4><p>{d}</p></div>')
        choice = st.radio("Assessment type", [t[2] for t in types], horizontal=True,
                          index=[t[2] for t in types].index(st.session_state.a_type), label_visibility="collapsed")
        b1, b2, _ = st.columns([1, 1, 5])
        b1.button("Back", on_click=go, args=("Assessment", 1), key="s2b")
        if b2.button("Continue", type="primary", key="s2c"):
            st.session_state.a_type = choice
            go("Assessment", 3)
            st.rerun()

    elif s == 3:
        stepper(3)
        st.markdown(f"### Record Your Child's Speech")
        st.caption("Follow the instructions and record the child's voice, or upload an existing recording.")
        tab_rec, tab_up = st.tabs(["🎙️ Record", "⬆️ Upload audio"])
        audio_bytes, aname = None, None
        with tab_rec:
            l, r = st.columns([1.6, 1])
            with l:
                task = st.selectbox("Task", ["Task 1: Word Repetition", "Task 2: Sentence Repetition",
                                             "Task 3: Picture Description", "Task 4: Free Speech"])
                prompts = {"Task 1": ("Say the word:", "“Apple”", "🍎"), "Task 2": ("Say:", "“The boy is playing.”", "🧒"),
                           "Task 3": ("Tell what is happening in the picture.", "🐶⚽", "🖼️"),
                           "Task 4": ("Talk about your favorite toy.", "🧸", "💬")}
                p = prompts[task[:6]]
                H(f'<div class="card" style="text-align:center"><small>{p[0]}</small>'
                  f'<div class="big-word">{p[1]}</div><div class="bigemoji">{p[2]}</div></div>')
                if hasattr(st, "audio_input"):
                    rec = st.audio_input("Ready to record?")
                    if rec:
                        audio_bytes, aname = rec.getvalue(), "recording.wav"
                else:
                    st.info("Your Streamlit version has no microphone widget. Upgrade to 1.40+ or use the Upload tab.")
            with r:
                H('''<div class="card"><b>Assessment Tasks</b>
                    <div class="task"><div class="n">1</div><div>Word Repetition<small>Say: Apple</small></div></div>
                    <div class="task"><div class="n">2</div><div>Sentence Repetition<small>Say: The boy is playing.</small></div></div>
                    <div class="task"><div class="n">3</div><div>Picture Description<small>Tell what is happening.</small></div></div>
                    <div class="task"><div class="n">4</div><div>Free Speech<small>Talk about your favorite toy.</small></div></div></div>''')
        with tab_up:
            l, r = st.columns([1.6, 1])
            with l:
                f = st.file_uploader("Drag & drop audio here", type=["wav", "mp3", "m4a"], help="Max 10 MB")
                if f:
                    if f.size > 10 * 1024 * 1024:
                        st.error("That file is larger than 10 MB. Choose a shorter recording.")
                    else:
                        audio_bytes, aname = f.getvalue(), f.name
                        st.audio(audio_bytes)
            with r:
                H('''<div class="card"><b>Audio Guidelines</b>
                    <div class="check"><span class="ok">✔</span>Clear speech recording</div>
                    <div class="check"><span class="ok">✔</span>Minimal background noise</div>
                    <div class="check"><span class="ok">✔</span>Use a quiet environment</div>
                    <div class="check"><span class="ok">✔</span>Record 30–60 seconds</div>
                    <div class="check"><span class="ok">✔</span>Child should speak naturally</div></div>''')
        b1, b2, _ = st.columns([1, 1.4, 4])
        b1.button("Back", on_click=go, args=("Assessment", 2), key="s3b")
        if b2.button("Analyze Speech", type="primary", key="s3c"):
            if audio_bytes:
                st.session_state.audio, st.session_state.audio_name = audio_bytes, aname
                go("Assessment", 4)
                st.rerun()
            else:
                st.warning("Record or upload an audio clip first.")

    elif s == 4:
        stepper(4)
        if st.session_state.results is None or st.session_state.results.get("_for") != zlib.crc32(st.session_state.audio):
            st.markdown("### Analyzing Speech…")
            st.caption("Our AI is processing the audio and extracting speech features.")
            steps = ["Audio uploaded", "Noise reduction", "Voice activity detection",
                     "Feature extraction (MFCC, pitch, formants, etc.)", "Speech analysis using TensorFlow (CNN + BiGRU)",
                     "Generating results…"]
            box, prog = st.empty(), st.progress(0)
            for i in range(len(steps)):
                box.markdown("".join(f'<div class="check"><span class="ok">{"✔" if j <= i else "○"}</span>{t}</div>'
                                     for j, t in enumerate(steps)), unsafe_allow_html=True)
                prog.progress(int((i + 1) / len(steps) * 100))
                time.sleep(.5)
            res = analyse_audio(st.session_state.audio, st.session_state.child)
            res["_for"] = zlib.crc32(st.session_state.audio)
            st.session_state.results = res
            st.rerun()
        results_view()


def results_view():
    res, child = st.session_state.results, st.session_state.child
    st.markdown("### Assessment Results")
    st.caption("Here is your child's speech and language development.")
    ov = res["overall"]
    col = {"green": "#22A06B", "orange": "#F59E0B", "red": "#E5484D"}[res["level"][1]]
    c1, c2, c3 = st.columns([1, 1.5, 1.2])
    with c1:
        H(f'''<div class="card" style="text-align:center"><b>Overall Score</b>
            <div class="gauge" style="background:conic-gradient({col} {ov*3.6}deg,#E8EEF9 0)"><div><b>{ov}%</b><span>Development Level</span></div></div>
            <span class="badge b-{res["level"][1]}">{res["level"][0]}</span></div>''')
    with c2:
        bars = "".join(f'<div class="bar-row"><span>{k}</span><div class="bar"><i style="width:{v}%;background:{bar_color(v)}"></i></div><span>{v}%</span></div>'
                       for k, v in res["cats"].items())
        H(f'<div class="card"><b>Category Scores</b>{bars}</div>')
    with c3:
        rows = "".join(f'<div class="metric-row"><span>{k}</span><b>{v}</b></div>' for k, v in res["metrics"].items())
        H(f'<div class="card"><b>Speech Metrics</b>{rows}</div>')
    H('<div style="height:.8rem"></div>')
    b1, b2, b3, b4 = st.columns(4)
    b1.download_button("Download Report", report_html(child, res), file_name="childvoice_report.html", mime="text/html", use_container_width=True)
    b2.button("View Detailed Report", on_click=go, args=("Report",), use_container_width=True, key="r_det")
    b3.button("For Parents", on_click=go, args=("Parents",), use_container_width=True, key="r_par")
    b4.button("New Assessment", type="primary", use_container_width=True, key="r_new",
              on_click=lambda: st.session_state.update(page="Assessment", step=1, results=None, audio=None))


def page_report():
    res, child = st.session_state.results, st.session_state.child
    if not res:
        st.info("Complete an assessment first to see a detailed report.")
        st.button("Start Assessment", type="primary", on_click=go, args=("Assessment", 1))
        return
    st.markdown("### AI Assessment Report")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        H(f'''<div class="card"><div class="metric-row"><span>Child Name</span><b>{child["name"]}</b></div>
            <div class="metric-row"><span>Age</span><b>{child["age"]} years</b></div>
            <div class="metric-row"><span>Assessment Date</span><b>{res["date"]}</b></div>
            <div class="metric-row"><span>Assessment Type</span><b>{st.session_state.a_type}</b></div>
            <div class="metric-row"><span>Overall Score</span><b>{res["overall"]}%</b></div>
            <div class="metric-row"><span>Development Level</span><b>{res["level"][0]}</b></div></div>''')
        strongest = max(res["cats"], key=res["cats"].get)
        weakest = min(res["cats"], key=res["cats"].get)
        st.markdown("#### Summary")
        st.write(f"The child's speech sample was analyzed using acoustic and language features. Overall performance "
                 f"falls in the **{res['level'][0].lower()}**. Strongest area: **{strongest}** ({res['cats'][strongest]}%). "
                 f"Area to keep an eye on: **{weakest}** ({res['cats'][weakest]}%).")
        st.markdown("#### Recommendations")
        st.markdown("- Encourage regular speaking and storytelling.\n- Use reading and conversation activities.\n"
                    "- Monitor fluency and vocabulary development.\n- Consider professional evaluation if concerns persist.")
    with c2:
        H('<div class="hero-art" style="height:200px">🧒📄</div>')
        H('<div class="disclaimer blue"><b>This is an AI-assisted screening report, not a medical diagnosis.</b></div>')
        st.download_button("Download Report", report_html(child, res), file_name="childvoice_report.html",
                           mime="text/html", type="primary", use_container_width=True)
        st.button("Back to Home", on_click=go, args=("Home",), use_container_width=True, key="rep_home")


def page_parents():
    H('<div class="hero" style="grid-template-columns:1.3fr 1fr"><div><h1>A Guide for Parents</h1>'
      '<p>Support your child\'s communication journey.</p></div>'
      '<div class="hero-art" style="height:200px"><div class="bubble">Talk • Play • Read • Grow</div>👩‍👧</div></div>')
    H('<div style="height:1rem"></div>')
    l, r = st.columns(2)
    with l:
        H('''<div class="card"><h4>What are we assessing?</h4>
            <div class="check">🗣️ Speech Clarity</div><div class="check">🔤 Pronunciation</div>
            <div class="check">📚 Vocabulary</div><div class="check">🌊 Fluency</div><div class="check">💬 Language Development</div></div>''')
    with r:
        H('''<div class="card"><h4>What do the results mean?</h4>
            <div class="task"><div class="n" style="background:#DDF6EA;color:#13744B">✔</div><div><b>Good Progress</b>
              <small>Your child's speech is within the expected range.</small></div></div>
            <div class="task"><div class="n" style="background:#FFEBCC;color:#935C00">!</div><div><b>Needs Observation</b>
              <small>Some characteristics may benefit from additional monitoring.</small></div></div>
            <div class="task"><div class="n" style="background:#FFE1E2;color:#B3262B">✚</div><div><b>Consider Professional Evaluation</b>
              <small>The screening identified characteristics that may benefit from assessment by a qualified professional.</small></div></div></div>''')
    H('<div class="disclaimer blue" style="text-align:center">💡 <b>Small Conversations Make a Big Difference.</b> Encourage. Listen. Support. Celebrate.</div>')


def page_professionals():
    st.markdown("### Professional Dashboard")
    st.caption("Monitor assessments and track progress.")
    cols = st.columns(4)
    for c, (n, l) in zip(cols, [("128", "Children Assessed"), ("42", "This Month"), ("76%", "Average Score"), ("12", "Needs Follow-up")]):
        c.markdown(f'<div class="stat"><b>{n}</b><span>{l}</span></div>', unsafe_allow_html=True)
    df = pd.DataFrame({"Child Name": ["Aarav S", "Diya M", "Kavin R", "Sana P", "Ishaan K"], "Age": [5, 6, 4, 5, 6],
                       "Speech": ["82%", "61%", "88%", "73%", "91%"], "Language": ["79%", "68%", "81%", "70%", "87%"],
                       "Status": ["Good", "Review", "Good", "Review", "Good"],
                       "Date": ["09 Sep 2026", "08 Sep 2026", "07 Sep 2026", "06 Sep 2026", "05 Sep 2026"]})
    st.markdown("#### Recent Assessments")
    st.dataframe(df, hide_index=True, use_container_width=True)
    trend = pd.DataFrame({"Avg score": [70, 72, 71, 74, 75, 76]}, index=["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    st.markdown("#### Average score trend")
    st.line_chart(trend, color="#1E6BFF", height=220)


def page_resources():
    H('<div class="hero" style="grid-template-columns:1.4fr 1fr"><div><h1>Learning Resources</h1>'
      '<p>Helpful information for parents, educators and professionals.</p></div>'
      '<div class="hero-art" style="height:180px"><div class="bubble">Learn • Support • Empower</div>👧</div></div>')
    H('<div style="height:1rem"></div>')
    feature_cards([("🧬", "i-purple", "Speech Development Milestones", "Age-wise speech and language development guide."),
                   ("🎲", "i-orange", "Fun Activities", "Games and activities to encourage speech at home.")])
    H('<div style="height:.8rem"></div>')
    feature_cards([("📖", "i-teal", "Reading Tips", "How to build vocabulary through reading."),
                   ("❓", "i-pink", "Common Questions", "Answers to frequently asked questions.")])


def page_contact():
    l, r = st.columns([1, 1.3])
    with l:
        st.markdown("### Contact Us")
        st.caption("We'd love to hear from you!")
        H('''<div class="card"><div class="task"><div class="n">✉</div><div>Email<small>info@childvoice.ai</small></div></div>
            <div class="task"><div class="n">☎</div><div>Phone<small>+91 98765 43210</small></div></div>
            <div class="task"><div class="n">📍</div><div>Location<small>Chennai, Tamil Nadu, India</small></div></div>
            <div class="task"><div class="n">👥</div><div>Project Team<small>ChildVoice AI Research Group</small></div></div></div>''')
    with r:
        with st.form("contact"):
            a, b = st.columns(2)
            a.text_input("Name")
            b.text_input("Email")
            st.text_input("Subject")
            st.text_area("Message", height=140)
            if st.form_submit_button("Send Message", type="primary"):
                st.success("Thanks! Your message has been sent. We'll reply within two working days.")


def page_login():
    l, r = st.columns([1.5, 1])
    with l:
        t1, t2 = st.tabs(["Login", "Register"])
        with t1:
            with st.form("login"):
                st.markdown("#### Welcome Back!")
                e = st.text_input("Email", placeholder="demo@childvoice.ai")
                p = st.text_input("Password", type="password", placeholder="demo123")
                if st.form_submit_button("Login", type="primary"):
                    if st.session_state.users.get(e) == p and p:
                        st.session_state.user = e
                        go("Home")
                        st.rerun()
                    else:
                        st.error("Email or password is incorrect.")
        with t2:
            with st.form("register"):
                st.markdown("#### Create an Account")
                n = st.text_input("Full Name")
                e = st.text_input("Email address")
                p = st.text_input("Password", type="password")
                c = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Register", type="primary"):
                    if not (n and e and p):
                        st.error("Fill in every field to register.")
                    elif p != c:
                        st.error("Passwords don't match.")
                    elif e in st.session_state.users:
                        st.error("That email already has an account. Log in instead.")
                    else:
                        st.session_state.users[e] = p
                        st.session_state.user = e
                        go("Home")
                        st.rerun()
        st.caption("Demo accounts are kept in session memory only. Connect MySQL for real storage.")
    with r:
        H('<div class="hero-art" style="height:340px"><div class="bubble">Different Voices,<br>Brighter Futures 💙</div>👩‍👧‍👦</div>')


def page_technology():
    st.markdown("### Our Technology")
    st.caption("Powered by advanced AI and deep learning.")
    items = [("🧠", "i-orange", "TensorFlow", "Deep learning model (CNN + BiGRU)"),
             ("🎛️", "i-purple", "Librosa", "Audio processing"),
             ("🗣️", "i-teal", "Whisper", "Speech-to-text (optional)"),
             ("🐍", "i-blue", "Python", "Backend and Streamlit app"),
             ("⚛️", "i-blue", "Streamlit + CSS", "Modern web interface"),
             ("🗄️", "i-orange", "MySQL", "Database"),
             ("📊", "i-pink", "Chart libraries", "Data visualization"),
             ("🔒", "i-green", "Secure & Private", "Your data is safe with us")]
    for row in (items[:4], items[4:]):
        feature_cards(row)
        H('<div style="height:.8rem"></div>')


PAGES = dict(Home=page_home, About=page_about, Assessment=page_assessment, Report=page_report, Parents=page_parents,
             Professionals=page_professionals, Resources=page_resources, Contact=page_contact, Login=page_login,
             Technology=page_technology)

nav()
PAGES[st.session_state.page]()
footer()
