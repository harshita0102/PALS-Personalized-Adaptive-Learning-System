import json
import os
import random
import re
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(BASE, "progress.json")
app = Flask(__name__)
LOCK = threading.Lock()

GOALS = ["Understand my mistakes", "Practise for a test", "Quick check"]
GOAL_NEXT = {
    "Understand my mistakes": "Redo the questions you got wrong and read each misconception note.",
    "Practise for a test": "Do one more set of five on this topic before the test.",
    "Quick check": "Good quick check. Come back when you want a deeper session.",
}

# Keyword fallback, used only if the trained classifier file cannot be loaded.
# Whole-word patterns: the old substring check matched "sign" inside "design" and
# "negative" inside "multiplying two negatives gives a negative answer" (a conceptual
# misconception), so it labelled that conceptual error as a calculation slip.
CALC_PATTERNS = [re.compile(p) for p in [
    r"\barithmetic", r"\bcalculat", r"\bcareless", r"\bslip\b", r"\bcopy",
    r"\bdecimal", r"\bwrong sign", r"\bnegative sign", r"\bsign error",
    r"\baddition error", r"\bmultiplication error",
]]

# Results from the report (Section 6) plus checks run on the uploaded dashboard data.
# If the Colab notebook is rerun and the numbers change, update them here.
EVIDENCE = {
    "dataset": {"answers": 2709647, "students": 27485, "skills": 265,
                "test_wrong_answers": 140622, "eedi_labelled": 9737},
    "next_answer": [
        {"model": "Logistic regression (baseline)", "short": "Baseline", "val_auc": None, "test_auc": 0.6598},
        {"model": "LSTM (DKT), selected", "short": "LSTM", "val_auc": 0.6966, "test_auc": 0.6941},
        {"model": "Transformer", "short": "Transformer", "val_auc": 0.6901, "test_auc": 0.6883},
        {"model": "Hybrid transformer-LSTM", "short": "Hybrid", "val_auc": 0.6946, "test_auc": 0.6929},
    ],
    "persistence": {"persistent_share": 0.417, "test_auc": 0.6747},
    "error_type": {"conceptual_share": 0.756, "test_auc": 0.7505, "f1": 0.8110,
                   "accuracy": 0.725, "majority_accuracy": 0.7676, "always_conceptual_f1": 0.8685},
    "hand_check": {"first_pass": [39, 60], "second_pass": [43, 60]},
    "forgetting": {"global_mse": 0.1948, "global_auc": 0.5631,
                   "hlr_mse": 0.1103, "hlr_auc": 0.8666, "mse_reduction_pct": 43.4},
    # Re-check on the 300-student dashboard sample, students 150-299 held out.
    # Simple models were fitted on students 0-149 only.
    "sample_check": {
        "stored_sequence_model_auc": 0.7197,
        "bkt_single_skill_auc": 0.6375,
        "history_logreg_auc": 0.6588,
        "running_mean_auc": 0.6562,
        "note": "The stored sequence-model predictions cover only these 300 students. "
                "The app does not run the LSTM or transformer; their weights are not in the upload.",
    },
}


def load_json(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


QUESTIONS = load_json("eedi_questions.json") or []
QBYID = {q["id"]: q for q in QUESTIONS}
TOPICS = {}
for q in QUESTIONS:
    TOPICS.setdefault(q["topic"], []).append(q["id"])
TOPICS = {t: ids for t, ids in TOPICS.items() if len(ids) >= 3}

DASH = load_json("dashboard_data.json") or {}
STUDENTS = DASH.get("students", {})

try:
    import joblib
    _model = joblib.load(os.path.join(BASE, "error_model.joblib"))
    VEC, CLF = _model["vec"], _model["clf"]
    CLASSIFIER_NOTE = "trained text classifier (TF-IDF + logistic regression)"
except Exception as exc:
    VEC, CLF = None, None
    CLASSIFIER_NOTE = f"keyword rule fallback ({type(exc).__name__})"


def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return {}
    try:
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_progress():
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(PROGRESS, f, indent=1)
    os.replace(tmp, PROGRESS_PATH)


PROGRESS = load_progress()
SESSIONS = {}


def student_key(name):
    return " ".join(name.lower().split())


def get_record(name):
    return PROGRESS.get(student_key(name))


def topic_mastery(rec, topic):
    """Beta-smoothed accuracy on a topic, with the number of answers behind it."""
    t = (rec or {}).get("topics", {}).get(topic, {"seen": 0, "correct": 0})
    return (t["correct"] + 1) / (t["seen"] + 2), t["seen"]


def classify(text, misconception_name):
    """Return P(conceptual) for a wrong answer and how it was estimated."""
    if CLF is not None:
        p = float(CLF.predict_proba(VEC.transform([text]))[0, 1])
        return p, "classifier"
    is_calc = any(p.search(str(misconception_name).lower()) for p in CALC_PATTERNS)
    return (0.0 if is_calc else 1.0), "keyword rule"


def pick_questions(topic, n, rec):
    """Choose n questions. If the student has a recurring misconception in this topic,
    put one question that tests it first; the rest are shuffled."""
    ids = list(TOPICS[topic])
    recurring = {m for m, c in (rec or {}).get("misconceptions", {}).items() if c >= 2}
    targeted = [i for i in ids
                if any(m in recurring for m in QBYID[i]["misconceptions"].values())]
    random.shuffle(targeted)
    rest = [i for i in ids if i not in targeted]
    random.shuffle(rest)
    return (targeted[:1] + rest)[:n]


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/api/status")
def status():
    needed = ["eedi_questions.json", "dashboard_data.json", "error_model.joblib"]
    missing = [f for f in needed if not os.path.exists(os.path.join(BASE, f))]
    return jsonify({"missing": missing, "classifier": CLASSIFIER_NOTE,
                    "topics": len(TOPICS), "students_with_progress": len(PROGRESS),
                    "sequence_model": "not loaded (weights not in upload)"})


@app.route("/api/topics")
def topics():
    return jsonify([{"name": t, "count": len(ids)} for t, ids in sorted(TOPICS.items())])


@app.route("/api/recommend")
def recommend():
    """Topics ranked by the student's weakest estimated mastery (lowest first)."""
    rec = get_record(request.args.get("name", ""))
    rows = []
    for topic, ids in TOPICS.items():
        m, seen = topic_mastery(rec, topic)
        rows.append({"topic": topic, "count": len(ids), "mastery": round(m, 3), "seen": seen})
    rows.sort(key=lambda r: (r["mastery"], r["seen"]))
    return jsonify(rows[:10])


@app.route("/api/progress")
def progress():
    rec = get_record(request.args.get("name", ""))
    if rec is None:
        return jsonify({"found": False})
    recurring = sorted(((m, c) for m, c in rec["misconceptions"].items() if c >= 2),
                       key=lambda x: -x[1])
    return jsonify({"found": True, "answers": len(rec["history"]),
                    "topics_practised": len(rec["topics"]),
                    "recurring_misconceptions": [{"text": m, "count": c} for m, c in recurring[:10]]})


@app.route("/api/session", methods=["POST"])
def create_session():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:40]
    topic = data.get("topic", "")
    goal = data.get("goal", "")
    try:
        n = int(data.get("n", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Choose a number of questions."}), 400
    if not name:
        return jsonify({"error": "Please enter your name."}), 400
    if topic not in TOPICS:
        return jsonify({"error": "Please choose a topic."}), 400
    if goal not in GOALS:
        return jsonify({"error": "Please choose a goal."}), 400
    n = max(1, min(n, len(TOPICS[topic]), 10))
    sid = uuid.uuid4().hex
    ids = pick_questions(topic, n, get_record(name))
    SESSIONS[sid] = {"name": name, "topic": topic, "goal": goal,
                     "ids": ids, "i": 0, "answers": []}
    return jsonify({"session": sid, "n": len(ids)})


@app.route("/api/question/<sid>")
def question(sid):
    s = SESSIONS.get(sid)
    if s is None:
        return jsonify({"error": "Session not found. Start again."}), 404
    if s["i"] >= len(s["ids"]):
        return jsonify({"done": True})
    q = QBYID[s["ids"][s["i"]]]
    return jsonify({"done": False, "index": s["i"] + 1, "total": len(s["ids"]),
                    "topic": s["topic"], "text": q["text"], "options": q["options"]})


@app.route("/api/answer/<sid>", methods=["POST"])
def answer(sid):
    s = SESSIONS.get(sid)
    if s is None:
        return jsonify({"error": "Session not found. Start again."}), 404
    if s["i"] >= len(s["ids"]):
        return jsonify({"error": "All questions are finished."}), 400
    if len(s["answers"]) > s["i"]:
        return jsonify(s["answers"][s["i"]])
    q = QBYID[s["ids"][s["i"]]]
    choice = (request.get_json(silent=True) or {}).get("choice", "")
    if choice not in q["options"]:
        return jsonify({"error": "Choose one of the options."}), 400

    correct = choice == q["correct"]
    result = {"correct": correct, "chosen": choice, "right_answer": q["correct"],
              "right_text": q["options"][q["correct"]]}
    name = ""
    with LOCK:
        rec = PROGRESS.setdefault(student_key(s["name"]),
                                  {"name": s["name"], "topics": {}, "misconceptions": {}, "history": []})
        if not correct:
            name = q["misconceptions"].get(choice, "")
            p, method = classify(f"{q['text']} || {q['options'][choice]}", name)
            prior = rec["misconceptions"].get(name, 0) if name else 0
            result.update({
                "kind": "conceptual" if p >= 0.5 else "computational",
                "p_conceptual": round(p, 3),
                # Close to 0.5 means the classifier is unsure; say so instead of hiding it.
                "confidence": "high" if abs(p - 0.5) >= 0.3 else "low",
                "method": method,
                "misconception": name,
                "recurring": prior >= 1,
            })
        t = rec["topics"].setdefault(s["topic"], {"seen": 0, "correct": 0})
        t["seen"] += 1
        t["correct"] += int(correct)
        if name:
            rec["misconceptions"][name] = rec["misconceptions"].get(name, 0) + 1
        rec["history"].append({"ts": int(time.time()), "topic": s["topic"],
                               "correct": correct, "misconception": name})
        rec["history"] = rec["history"][-500:]
        save_progress()

    s["answers"].append(result)
    return jsonify(result)


@app.route("/api/next/<sid>", methods=["POST"])
def next_question(sid):
    s = SESSIONS.get(sid)
    if s is None:
        return jsonify({"error": "Session not found. Start again."}), 404
    if len(s["answers"]) > s["i"]:
        s["i"] += 1
    return jsonify({"done": s["i"] >= len(s["ids"])})


@app.route("/api/summary/<sid>")
def summary(sid):
    s = SESSIONS.get(sid)
    if s is None:
        return jsonify({"error": "Session not found. Start again."}), 404
    if len(s["answers"]) < len(s["ids"]):
        return jsonify({"error": "Finish all questions first."}), 400
    counts = {"correct": 0, "conceptual": 0, "computational": 0}
    items = []
    for i, a in enumerate(s["answers"], 1):
        if a["correct"]:
            counts["correct"] += 1
            status_text = "Correct"
        elif a["kind"] == "conceptual":
            counts["conceptual"] += 1
            status_text = "Conceptual gap (estimate)"
        else:
            counts["computational"] += 1
            status_text = "Calculation slip (estimate)"
        items.append({"index": i, "chosen": a["chosen"], "status": status_text,
                      "misconception": a.get("misconception", ""),
                      "recurring": a.get("recurring", False),
                      "confidence": a.get("confidence", "")})
    wrong = counts["conceptual"] + counts["computational"]
    if wrong == 0:
        advice = "No mistakes this round. Try a harder topic or more questions next time."
    elif counts["conceptual"] >= counts["computational"]:
        advice = (f"Most of your mistakes look like gaps in understanding. "
                  f"Spend time on the explanation of {s['topic']} before more practice.")
    else:
        advice = ("Most of your mistakes look like slips in the working. "
                  "Slow down, write each step, and check signs and arithmetic.")

    recurring_now = sorted({a["misconception"] for a in s["answers"]
                            if a.get("recurring") and a.get("misconception")})
    if recurring_now:
        advice += (" You have made this kind of mistake before: "
                   + "; ".join(recurring_now[:3]) + ". Repeat questions on it until it stops coming up.")

    with LOCK:
        rec = get_record(s["name"]) or {}
        mastery, seen = topic_mastery(rec, s["topic"])

    return jsonify({"name": s["name"], "topic": s["topic"], "goal": s["goal"],
                    "total": len(s["ids"]), "counts": counts, "items": items,
                    "advice": advice, "goal_next": GOAL_NEXT[s["goal"]],
                    "recurring_misconceptions": recurring_now,
                    "topic_mastery": round(mastery, 3), "topic_answers_total": seen})


@app.route("/api/evidence")
def evidence():
    return jsonify(EVIDENCE)


@app.route("/api/trace-ids")
def trace_ids():
    return jsonify(list(STUDENTS.keys())[:300])


@app.route("/api/trace/<sid>")
def trace(sid):
    t = STUDENTS.get(str(sid))
    if t is None:
        return jsonify({"error": "Student not found."}), 404
    return jsonify(t)


if __name__ == "__main__":
    app.run(debug=True)