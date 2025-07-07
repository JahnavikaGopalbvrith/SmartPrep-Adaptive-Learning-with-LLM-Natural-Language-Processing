from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_session import Session
import os
import time
from groq import Groq
from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone as PineconeVector
from langchain.embeddings import HuggingFaceEmbeddings
from graphviz import Digraph
from newspaper import Article
import re
import json


# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "flatg"

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index = pinecone.Index(PINECONE_INDEX)

# Flask App
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "your_secret_key"
Session(app)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVector(index=index, embedding_function=embeddings.embed_query, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

@app.route("/")
def home():
    return render_template("index.html")

# --- DAA Routes ---

@app.route("/daa")
def daa_learn():
    return render_template("daa.html")

@app.route("/daa_ask", methods=["POST"])
def daa_ask_question():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if "daa_conversation_history" not in session:
        session["daa_conversation_history"] = []

    conversation_history = session["daa_conversation_history"]
    conversation_history.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an expert tutor in DAA. Answer questions based on your knowledge."},
            {"role": "user", "content": question}
        ]
    )

    if response and response.choices and response.choices[0].message:
        answer = response.choices[0].message.content.strip()
    else:
        answer = "Sorry, I couldn't generate an answer."

    conversation_history.append({"role": "assistant", "content": answer})
    session["daa_conversation_history"] = conversation_history  # Correct key
    session.modified = True

    return jsonify({"answer": answer})

import json
import re

@app.route("/daa_quiz", methods=["GET", "POST"])
def daa_quiz_intro():
    if request.method == "POST":
        difficulty = request.form.get("difficulty")
        num_questions = int(request.form.get("num_questions"))

        conversation_history = session.get("daa_conversation_history", [])
        if not conversation_history:
            return render_template("daa_quiz_intro.html", error="Please study first by asking questions.")

        user_questions = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        topics = "\n".join(user_questions)

        prompt = f"""
You are an expert tutor for Design and Analysis of Algorithms (DAA).

Based on the user's study history below, generate {num_questions} multiple-choice questions of {difficulty} difficulty.

Each question must be formatted as a JSON object with exactly these fields:
- "question": the question text.
- "options": a list of exactly 4 strings.
- "answer": the correct option text (must be exactly one of the 4 options).

Return ONLY a JSON array of question objects. Do not include any explanation, markdown, or formatting.

Study history:
{topics}
"""

        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an AI tutor for DAA."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw_output = response.choices[0].message.content

            # Extract valid JSON array even if wrapped in markdown
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", raw_output, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found.")

            json_data = json_match.group(0)
            questions = json.loads(json_data)

            # Basic structure validation
            for q in questions:
                assert isinstance(q["question"], str)
                assert isinstance(q["options"], list) and len(q["options"]) == 4
                assert q["answer"] in q["options"]

        except Exception as e:
            return render_template("daa_quiz_intro.html", error="Failed to parse quiz. Please try again.")

        session["daa_quiz_questions"] = questions
        session["daa_quiz_index"] = 0
        session["daa_quiz_score"] = 0
        session["daa_wrong_answers"] = []
        session.modified = True

        return redirect(url_for("daa_quiz_question"))

    return render_template("daa_quiz_intro.html")
@app.route("/daa_quiz_question", methods=["GET", "POST"])
def daa_quiz_question():
    if "daa_quiz_questions" not in session:
        return redirect(url_for("daa_quiz_intro"))

    questions = session["daa_quiz_questions"]
    index = session["daa_quiz_index"]

    if request.method == "POST":
        selected_index = int(request.form.get("answer", -1))
        current_question = questions[index]

        if 0 <= selected_index < 4:
            selected_option = current_question["options"][selected_index]
            correct_option = current_question["answer"]

            if selected_option == correct_option:
                session["daa_quiz_score"] += 1
            else:
                session["daa_wrong_answers"].append({
                    "question": current_question["question"],
                    "your_answer": selected_option,
                    "correct_answer": correct_option
                })

        session["daa_quiz_index"] += 1
        session.modified = True

        if session["daa_quiz_index"] >= len(questions):
            return redirect(url_for("daa_quiz_result"))

        return redirect(url_for("daa_quiz_question"))

    current_question = questions[index]
    return render_template("daa_quiz_question.html", index=index + 1, question=current_question)
@app.route("/daa_quiz_result")
def daa_quiz_result():
    score = session.get("daa_quiz_score", 0)
    total = len(session.get("daa_quiz_questions", []))
    wrong_answers = session.get("daa_wrong_answers", [])

    session.clear()
    return render_template("daa_quiz_result.html", score=score, total=total, wrong_answers=wrong_answers)

# --- FLAT Routes ---

@app.route("/flat")
def flat_learn():
    return render_template("flat.html")

@app.route("/flat_visualize_transition")
def flat_visualize_transition():
    return render_template("flat_visualize.html")
@app.route("/flat_ask", methods=["POST"])
def flat_ask_question():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if "flat_conversation_history" not in session:
        session["flat_conversation_history"] = []
    conversation_history = session["flat_conversation_history"]

    system_prompt = """
You are an expert tutor specialized in Formal Languages and Automata Theory (FLAT).
Answer questions clearly and accurately based on the provided context.
Always format your answers using Markdown.
"""

    conversation_history.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": question}]
    )

    if response and response.choices and response.choices[0].message:
        answer = response.choices[0].message.content.strip()
    else:
        answer = "Sorry, I couldn't generate an answer."

    conversation_history.append({"role": "assistant", "content": answer})
    session["flat_conversation_history"] = conversation_history  # FIXED here
    session.modified = True

    return jsonify({"answer": answer})

# --- FLAT Quiz Flow ---
@app.route("/flat_quiz", methods=["GET", "POST"])
def flat_quiz_intro():
    if request.method == "POST":
        difficulty = request.form.get("difficulty")
        num_questions = int(request.form.get("num_questions"))

        conversation_history = session.get("flat_conversation_history", [])
        if not conversation_history:
            return render_template("flat_quiz_intro.html", error="Please study first by asking questions.")

        user_questions = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        topics = "\n".join(user_questions)

        prompt = f"""
You are an expert tutor for Formal Language and Automata Theory (FLAT).

Based on the user's study history below, generate {num_questions} multiple-choice questions of {difficulty} difficulty.

Each question must be formatted as a JSON object with exactly these fields:
- "question": the question text.
- "options": a list of exactly 4 strings.
- "answer": the correct option text (must be exactly one of the 4 options).

Return ONLY a JSON array of question objects. Do not include any explanation, markdown, or formatting.

Study history:
{topics}
"""

        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an AI tutor for FLAT."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw_output = response.choices[0].message.content

            # Extract valid JSON array even if wrapped in markdown
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", raw_output, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found.")

            json_data = json_match.group(0)
            questions = json.loads(json_data)

            # Basic structure validation
            for q in questions:
                assert isinstance(q["question"], str)
                assert isinstance(q["options"], list) and len(q["options"]) == 4
                assert q["answer"] in q["options"]

        except Exception as e:
            return render_template("flat_quiz_intro.html", error="Failed to parse quiz. Please try again.")

        session["flat_quiz_questions"] = questions
        session["flat_quiz_index"] = 0
        session["flat_quiz_score"] = 0
        session["flat_wrong_answers"] = []
        session.modified = True

        return redirect(url_for("flat_quiz_question"))

    return render_template("flat_quiz_intro.html")


@app.route("/flat_quiz_question", methods=["GET", "POST"])
def flat_quiz_question():
    if "flat_quiz_questions" not in session:
        return redirect(url_for("flat_quiz_intro"))

    questions = session["flat_quiz_questions"]
    index = session["flat_quiz_index"]

    if request.method == "POST":
        selected_index = int(request.form.get("answer", -1))
        current_question = questions[index]

        if 0 <= selected_index < 4:
            selected_option = current_question["options"][selected_index]
            correct_option = current_question["answer"]

            if selected_option == correct_option:
                session["flat_quiz_score"] += 1
            else:
                session["flat_wrong_answers"].append({
                    "question": current_question["question"],
                    "your_answer": selected_option,
                    "correct_answer": correct_option
                })

        session["flat_quiz_index"] += 1
        session.modified = True

        if session["flat_quiz_index"] >= len(questions):
            return redirect(url_for("flat_quiz_result"))

        return redirect(url_for("flat_quiz_question"))

    current_question = questions[index]
    return render_template("flat_quiz_question.html", index=index + 1, question=current_question)


@app.route("/flat_quiz_result")
def flat_quiz_result():
    score = session.get("flat_quiz_score", 0)
    total = len(session.get("flat_quiz_questions", []))
    wrong_answers = session.get("flat_wrong_answers", [])

    session.clear()
    return render_template("flat_quiz_result.html", score=score, total=total, wrong_answers=wrong_answers)

# --- Aptitude Routes ---

@app.route("/aptitude")
def aptitude_learn():
    return render_template("aptitude.html")


@app.route("/aptitude_ask", methods=["POST"])
def aptitude_ask_question():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if "aptitude_conversation_history" not in session:
        session["aptitude_conversation_history"] = []

    conversation_history = session["aptitude_conversation_history"]
    conversation_history.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an expert tutor in Aptitude. Answer questions concisely."},
            {"role": "user", "content": question}
        ]
    )

    if response and response.choices and response.choices[0].message:
        answer = response.choices[0].message.content.strip()
    else:
        answer = "Sorry, I couldn't generate an answer."

    conversation_history.append({"role": "assistant", "content": answer})
    session["aptitude_conversation_history"] = conversation_history
    session.modified = True

    return jsonify({"answer": answer})
@app.route("/aptitude_quiz", methods=["GET", "POST"])

@app.route("/aptitude_quiz", methods=["GET", "POST"])
def aptitude_quiz_intro():
    if request.method == "POST":
        difficulty = request.form.get("difficulty")
        num_questions = int(request.form.get("num_questions"))

        conversation_history = session.get("aptitude_conversation_history", [])
        if not conversation_history:
            return render_template("aptitude_quiz_intro.html", error="Please study first by asking questions.")

        user_questions = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        topics = "\n".join(user_questions)

        prompt = f"""
You are an expert tutor for Aptitude.

Based on the user's study history below, generate {num_questions} multiple-choice questions of {difficulty} difficulty.

Each question must be formatted as a JSON object with exactly these fields:
- "question": the question text.
- "options": a list of exactly 4 strings.
- "answer": the correct option text (must be exactly one of the 4 options).

Return ONLY a JSON array of question objects. Do not include any explanation, markdown, or formatting.

Study history:
{topics}
"""

        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an AI tutor for Aptitude."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw_output = response.choices[0].message.content

            # Extract valid JSON array even if wrapped in markdown
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", raw_output, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found.")

            json_data = json_match.group(0)
            questions = json.loads(json_data)

            # Basic structure validation
            for q in questions:
                assert isinstance(q["question"], str)
                assert isinstance(q["options"], list) and len(q["options"]) == 4
                assert q["answer"] in q["options"]

        except Exception as e:
            return render_template("aptitude_quiz_intro.html", error="Failed to parse quiz. Please try again.")

        session["aptitude_quiz_questions"] = questions
        session["aptitude_quiz_index"] = 0
        session["aptitude_quiz_score"] = 0
        session["aptitude_wrong_answers"] = []
        session.modified = True

        return redirect(url_for("aptitude_quiz_question"))

    return render_template("aptitude_quiz_intro.html")


@app.route("/aptitude_quiz_question", methods=["GET", "POST"])
def aptitude_quiz_question():
    if "aptitude_quiz_questions" not in session:
        return redirect(url_for("aptitude_quiz_intro"))

    questions = session["aptitude_quiz_questions"]
    index = session["aptitude_quiz_index"]

    if request.method == "POST":
        selected_index = int(request.form.get("answer", -1))
        current_question = questions[index]

        if 0 <= selected_index < 4:
            selected_option = current_question["options"][selected_index]
            correct_option = current_question["answer"]

            if selected_option == correct_option:
                session["aptitude_quiz_score"] += 1
            else:
                session["aptitude_wrong_answers"].append({
                    "question": current_question["question"],
                    "your_answer": selected_option,
                    "correct_answer": correct_option
                })

        session["aptitude_quiz_index"] += 1
        session.modified = True

        if session["aptitude_quiz_index"] >= len(questions):
            return redirect(url_for("aptitude_quiz_result"))

        return redirect(url_for("aptitude_quiz_question"))

    current_question = questions[index]
    return render_template("aptitude_quiz_question.html", index=index + 1, question=current_question)


@app.route("/aptitude_quiz_result")
def aptitude_quiz_result():
    score = session.get("aptitude_quiz_score", 0)
    total = len(session.get("aptitude_quiz_questions", []))
    wrong_answers = session.get("aptitude_wrong_answers", [])

    session.clear()
    return render_template("aptitude_quiz_result.html", score=score, total=total, wrong_answers=wrong_answers)


def summarize_text(text: str) -> str:
    system_prompt = (
        "You are an expert assistant who summarizes articles concisely "
        "in clear bullet points."
    )
    question = f"Here is the content of the article :\n\n{text} . Understand it and summarize it in your own words in simple language."

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        # Access the content safely
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        return f"Error generating summary: {e}"


@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    summary = None
    error = None

    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        if url:
            try:
                article = Article(url)
                article.download()
                article.parse()
                article_text = article.text

                summary = summarize_text(article_text)
            except Exception as e:
                error = f"Failed to process the URL. Error: {e}"
        else:
            error = "Please provide a valid URL."

    return render_template('summarize.html', summary=summary, error=error)



if __name__ == "__main__":
    app.run(debug=True)
