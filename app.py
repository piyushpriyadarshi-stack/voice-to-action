import streamlit as st
import sqlite3
import json
import hashlib
import tempfile
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime
from io import BytesIO

from faster_whisper import WhisperModel

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

DATABASE = "voice2action.db"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Voice2Action",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DATABASE)


def init_database():

    connection = get_connection()
    cursor = connection.cursor()

    # USERS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    # MEETINGS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            transcript TEXT,
            summary TEXT,
            key_points TEXT,
            action_items TEXT,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


init_database()


# =========================================================
# PASSWORD HASH
# =========================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()


# =========================================================
# REGISTER USER
# =========================================================

def register_user(username, password):

    connection = get_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (username, password)
            VALUES (?, ?)
            """,
            (
                username,
                hash_password(password)
            )
        )

        connection.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        connection.close()


# =========================================================
# LOGIN USER
# =========================================================

def login_user(username, password):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, username
        FROM users
        WHERE username = ?
        AND password = ?
        """,
        (
            username,
            hash_password(password)
        )
    )

    result = cursor.fetchone()

    connection.close()

    return result


# =========================================================
# SAVE MEETING
# =========================================================

def save_meeting(
    user_id,
    filename,
    transcript,
    summary,
    key_points,
    action_items
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO meetings
        (
            user_id,
            filename,
            transcript,
            summary,
            key_points,
            action_items,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            filename,
            transcript,
            summary,
            json.dumps(key_points),
            json.dumps(action_items),
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    connection.commit()
    connection.close()


# =========================================================
# GET USER MEETINGS
# =========================================================

def get_user_meetings(user_id):

    connection = get_connection()

    query = """
        SELECT
            id,
            filename,
            created_at
        FROM meetings
        WHERE user_id = ?
        ORDER BY id DESC
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=(user_id,)
    )

    connection.close()

    return dataframe


# =========================================================
# GET MEETING
# =========================================================

def get_meeting(meeting_id, user_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM meetings
        WHERE id = ?
        AND user_id = ?
        """,
        (
            meeting_id,
            user_id
        )
    )

    result = cursor.fetchone()

    connection.close()

    return result


# =========================================================
# DELETE MEETING
# =========================================================

def delete_meeting(meeting_id, user_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM meetings
        WHERE id = ?
        AND user_id = ?
        """,
        (
            meeting_id,
            user_id
        )
    )

    connection.commit()
    connection.close()


# =========================================================
# UPDATE TASK STATUS
# =========================================================

def update_task_status(
    meeting_id,
    user_id,
    task_index,
    new_status
):

    meeting = get_meeting(
        meeting_id,
        user_id
    )

    if not meeting:
        return False

    tasks = json.loads(
        meeting[6]
    )

    if task_index < 0 or task_index >= len(tasks):
        return False

    tasks[task_index]["status"] = new_status

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE meetings
        SET action_items = ?
        WHERE id = ?
        AND user_id = ?
        """,
        (
            json.dumps(tasks),
            meeting_id,
            user_id
        )
    )

    connection.commit()
    connection.close()

    return True


# =========================================================
# WHISPER MODEL
# =========================================================

@st.cache_resource
def load_whisper():

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


# =========================================================
# TRANSCRIBE AUDIO
# =========================================================

def transcribe_audio(audio_path):

    model = load_whisper()

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    transcript = ""

    for segment in segments:

        transcript += (
            segment.text.strip()
            + " "
        )

    return transcript.strip()


# =========================================================
# CHECK OLLAMA
# =========================================================

def check_ollama():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        if response.status_code == 200:

            models = response.json().get(
                "models",
                []
            )

            model_names = [
                model.get("name", "")
                for model in models
            ]

            if (
                OLLAMA_MODEL in model_names
                or any(
                    name.startswith(
                        OLLAMA_MODEL + ":"
                    )
                    for name in model_names
                )
            ):

                return True, "Ollama and model are ready."

            return True, (
                f"Ollama is running, but "
                f"{OLLAMA_MODEL} was not found."
            )

        return False, "Ollama is not responding correctly."

    except requests.exceptions.ConnectionError:

        return False, (
            "Ollama is not running. "
            "Start Ollama first."
        )

    except Exception as e:

        return False, str(e)


# =========================================================
# OLLAMA AI ANALYSIS
# =========================================================

def analyze_with_ollama(transcript):

    prompt = f"""
You are Voice2Action, an AI meeting assistant.

Analyze the following meeting transcript.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "summary": "Short and clear meeting summary",

    "key_points": [
        "Important point 1",
        "Important point 2"
    ],

    "action_items": [
        {{
            "task": "Task description",
            "assigned_to": "Person or Not specified",
            "deadline": "Deadline or Not specified",
            "priority": "High, Medium, Low, or Not specified",
            "status": "Pending"
        }}
    ]
}}

IMPORTANT RULES:

1. Do not invent information.
2. Only use information present in the transcript.
3. If an owner is not mentioned, use "Not specified".
4. If a deadline is not mentioned, use "Not specified".
5. If priority is not mentioned, use "Not specified".
6. Status must initially be "Pending".
7. Return valid JSON only.
"""

    data = {

        "model": OLLAMA_MODEL,

        "messages": [

            {
                "role": "system",
                "content": (
                    "You are a professional "
                    "meeting analysis assistant."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        "stream": False
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=data,
            timeout=300
        )

        if response.status_code != 200:

            return None, response.text

        result = response.json()

        if "message" not in result:

            return None, (
                "Unexpected Ollama response."
            )

        content = result[
            "message"
        ].get(
            "content",
            ""
        )

        return content, None

    except requests.exceptions.ConnectionError:

        return None, (
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return None, (
            "Ollama took too long to respond."
        )

    except Exception as e:

        return None, str(e)


# =========================================================
# PARSE AI JSON
# =========================================================

def parse_ai_response(response):

    if not response:
        return None

    try:

        return json.loads(response)

    except json.JSONDecodeError:

        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    response[
                        start:end + 1
                    ]
                )

            except json.JSONDecodeError:

                return None

    return None


# =========================================================
# NORMALIZE ANALYSIS
# =========================================================

def normalize_analysis(analysis):

    if not isinstance(
        analysis,
        dict
    ):
        return None

    summary = analysis.get(
        "summary",
        "Not available"
    )

    key_points = analysis.get(
        "key_points",
        []
    )

    action_items = analysis.get(
        "action_items",
        []
    )

    if not isinstance(
        key_points,
        list
    ):
        key_points = []

    if not isinstance(
        action_items,
        list
    ):
        action_items = []

    normalized_tasks = []

    for task in action_items:

        if not isinstance(
            task,
            dict
        ):
            continue

        normalized_tasks.append(
            {
                "task": task.get(
                    "task",
                    "Not specified"
                ),

                "assigned_to": task.get(
                    "assigned_to",
                    "Not specified"
                ),

                "deadline": task.get(
                    "deadline",
                    "Not specified"
                ),

                "priority": task.get(
                    "priority",
                    "Not specified"
                ),

                "status": task.get(
                    "status",
                    "Pending"
                )
            }
        )

    return {
        "summary": summary,
        "key_points": key_points,
        "action_items": normalized_tasks
    }


# =========================================================
# PDF GENERATION
# =========================================================

def create_pdf(
    filename,
    summary,
    key_points,
    action_items
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Voice2Action Meeting Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            f"Meeting: {filename}",
            styles["Heading2"]
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    elements.append(
        Paragraph(
            "Summary",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            summary,
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 15)
    )

    elements.append(
        Paragraph(
            "Key Points",
            styles["Heading2"]
        )
    )

    for point in key_points:

        elements.append(
            Paragraph(
                "• " + str(point),
                styles["BodyText"]
            )
        )

    elements.append(
        Spacer(1, 15)
    )

    elements.append(
        Paragraph(
            "Action Items",
            styles["Heading2"]
        )
    )

    table_data = [

        [
            "Task",
            "Assigned To",
            "Deadline",
            "Priority",
            "Status"
        ]

    ]

    for item in action_items:

        table_data.append(

            [
                item.get(
                    "task",
                    "Not specified"
                ),

                item.get(
                    "assigned_to",
                    "Not specified"
                ),

                item.get(
                    "deadline",
                    "Not specified"
                ),

                item.get(
                    "priority",
                    "Not specified"
                ),

                item.get(
                    "status",
                    "Pending"
                )
            ]

        )

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(

        TableStyle(

            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.grey
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.black
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]

        )

    )

    elements.append(table)

    document.build(elements)

    buffer.seek(0)

    return buffer


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

if "user_id" not in st.session_state:

    st.session_state.user_id = None

if "username" not in st.session_state:

    st.session_state.username = None

if "analysis" not in st.session_state:

    st.session_state.analysis = None

if "last_transcript" not in st.session_state:

    st.session_state.last_transcript = None

if "last_filename" not in st.session_state:

    st.session_state.last_filename = None


# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.title("🎙️ Voice2Action")

    st.subheader(
        "Local AI Meeting Assistant"
    )

    st.info(
        "Free architecture: "
        "Faster-Whisper + Ollama + SQLite"
    )

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Register"
        ]
    )

    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "🔐 Login",
            use_container_width=True
        ):

            user = login_user(
                username,
                password
            )

            if user:

                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    # -----------------------------------------------------
    # REGISTER
    # -----------------------------------------------------

    with register_tab:

        new_username = st.text_input(
            "Create Username"
        )

        new_password = st.text_input(
            "Create Password",
            type="password"
        )

        if st.button(
            "📝 Create Account",
            use_container_width=True
        ):

            if not new_username or not new_password:

                st.warning(
                    "Enter username and password."
                )

            elif register_user(
                new_username,
                new_password
            ):

                st.success(
                    "Account created successfully."
                )

            else:

                st.error(
                    "Username already exists."
                )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎙️ Voice2Action")

    st.caption(
        f"Logged in as: "
        f"{st.session_state.username}"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🎵 Analyze Meeting",
            "📚 Meeting History",
            "✅ Tasks",
            "⚙️ System Status"
        ]
    )

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None

        st.session_state.analysis = None
        st.session_state.last_transcript = None

        st.rerun()


# =========================================================
# LOAD MEETINGS
# =========================================================

meetings = get_user_meetings(
    st.session_state.user_id
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.title("📊 Dashboard")

    st.write(
        f"Welcome, **{st.session_state.username}** 👋"
    )

    total_meetings = len(meetings)

    total_tasks = 0
    completed_tasks = 0
    pending_tasks = 0

    for _, row in meetings.iterrows():

        meeting = get_meeting(
            int(row["id"]),
            st.session_state.user_id
        )

        if meeting:

            try:

                tasks = json.loads(
                    meeting[6]
                )

            except:

                tasks = []

            total_tasks += len(tasks)

            for task in tasks:

                if task.get(
                    "status"
                ) == "Completed":

                    completed_tasks += 1

                else:

                    pending_tasks += 1


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎙️ Meetings",
            total_meetings
        )

    with col2:

        st.metric(
            "✅ Total Tasks",
            total_tasks
        )

    with col3:

        st.metric(
            "⏳ Pending",
            pending_tasks
        )

    with col4:

        st.metric(
            "🏆 Completed",
            completed_tasks
        )


    st.divider()

    st.subheader(
        "🤖 Local AI Engine"
    )

    ollama_ok, ollama_message = check_ollama()

    if ollama_ok:

        st.success(
            f"🟢 {ollama_message}"
        )

    else:

        st.error(
            f"🔴 {ollama_message}"
        )


    st.divider()

    st.subheader(
        "🕒 Recent Meetings"
    )

    if meetings.empty:

        st.info(
            "No meetings yet. "
            "Analyze your first meeting."
        )

    else:

        st.dataframe(
            meetings.head(10),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ANALYZE MEETING
# =========================================================

elif page == "🎵 Analyze Meeting":

    st.title(
        "🎵 Analyze New Meeting"
    )

    st.write(
        "Upload a meeting recording and "
        "Voice2Action will convert it into "
        "structured information."
    )

    st.info(
        "Supported: WAV, MP3, M4A, FLAC, OGG"
    )

    uploaded_file = st.file_uploader(
        "Upload audio",
        type=[
            "wav",
            "mp3",
            "m4a",
            "flac",
            "ogg"
        ]
    )

    if uploaded_file:

        st.success(
            f"Audio loaded: {uploaded_file.name}"
        )

        st.audio(
            uploaded_file
        )

        meeting_title = st.text_input(
            "Meeting title",
            value=Path(
                uploaded_file.name
            ).stem
        )

        if st.button(
            "🚀 Analyze Meeting",
            type="primary",
            use_container_width=True
        ):

            extension = Path(
                uploaded_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as temp:

                temp.write(
                    uploaded_file.read()
                )

                audio_path = temp.name


            # -------------------------------------------------
            # WHISPER
            # -------------------------------------------------

            with st.spinner(
                "🎤 Transcribing audio..."
            ):

                try:

                    transcript = transcribe_audio(
                        audio_path
                    )

                except Exception as e:

                    st.error(
                        f"Transcription failed: {e}"
                    )

                    st.stop()


            if not transcript:

                st.error(
                    "No speech detected."
                )

                st.stop()


            st.success(
                "🎤 Transcription completed."
            )


            # -------------------------------------------------
            # OLLAMA
            # -------------------------------------------------

            with st.spinner(
                "🤖 Local AI analyzing meeting..."
            ):

                response, error = analyze_with_ollama(
                    transcript
                )


            if error:

                st.error(
                    error
                )

                st.info(
                    "Make sure Ollama is running "
                    "and Llama 3.2 is installed."
                )

                st.stop()


            # -------------------------------------------------
            # PARSE
            # -------------------------------------------------

            analysis = parse_ai_response(
                response
            )

            analysis = normalize_analysis(
                analysis
            )


            if not analysis:

                st.error(
                    "AI returned invalid JSON."
                )

                with st.expander(
                    "Show AI response"
                ):

                    st.code(
                        response
                    )

                st.stop()


            # SAVE TO SESSION
            st.session_state.analysis = analysis
            st.session_state.last_transcript = transcript
            st.session_state.last_filename = (
                meeting_title
            )


            # -------------------------------------------------
            # SAVE DATABASE
            # -------------------------------------------------

            save_meeting(

                st.session_state.user_id,

                meeting_title,

                transcript,

                analysis["summary"],

                analysis["key_points"],

                analysis["action_items"]

            )


            st.success(
                "💾 Meeting saved successfully."
            )


    # ---------------------------------------------------------
    # DISPLAY LAST ANALYSIS
    # ---------------------------------------------------------

    if (
        st.session_state.analysis
        and st.session_state.last_transcript
    ):

        analysis = st.session_state.analysis

        transcript = (
            st.session_state.last_transcript
        )

        filename = (
            st.session_state.last_filename
        )


        st.divider()

        st.header(
            "📝 Transcript"
        )

        with st.expander(
            "View full transcript",
            expanded=False
        ):

            st.write(
                transcript
            )


        st.header(
            "📋 Summary"
        )

        st.info(
            analysis["summary"]
        )


        st.header(
            "💡 Key Points"
        )

        if analysis["key_points"]:

            for point in analysis["key_points"]:

                st.write(
                    f"✅ {point}"
                )

        else:

            st.write(
                "No key points detected."
            )


        st.header(
            "✅ Action Items"
        )

        tasks = analysis[
            "action_items"
        ]

        if tasks:

            task_table = []

            for item in tasks:

                task_table.append(

                    {
                        "Task":
                            item.get(
                                "task",
                                "Not specified"
                            ),

                        "Assigned To":
                            item.get(
                                "assigned_to",
                                "Not specified"
                            ),

                        "Deadline":
                            item.get(
                                "deadline",
                                "Not specified"
                            ),

                        "Priority":
                            item.get(
                                "priority",
                                "Not specified"
                            ),

                        "Status":
                            item.get(
                                "status",
                                "Pending"
                            )
                    }

                )

            st.dataframe(
                pd.DataFrame(task_table),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No action items detected."
            )


        # -----------------------------------------------------
        # DOWNLOAD
        # -----------------------------------------------------

        st.divider()

        st.subheader(
            "📥 Export"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.download_button(

                "📝 Transcript",

                transcript,

                file_name=(
                    "transcript.txt"
                ),

                mime="text/plain",

                use_container_width=True
            )


        with col2:

            json_data = json.dumps(
                analysis,
                indent=4
            )

            st.download_button(

                "🧠 AI Analysis",

                json_data,

                file_name=(
                    "meeting_analysis.json"
                ),

                mime="application/json",

                use_container_width=True
            )


        with col3:

            pdf = create_pdf(

                filename,

                analysis["summary"],

                analysis["key_points"],

                analysis["action_items"]

            )

            st.download_button(

                "📄 PDF Report",

                pdf,

                file_name=(
                    "meeting_report.pdf"
                ),

                mime="application/pdf",

                use_container_width=True
            )


# =========================================================
# MEETING HISTORY
# =========================================================

elif page == "📚 Meeting History":

    st.title(
        "📚 Meeting History"
    )

    meetings = get_user_meetings(
        st.session_state.user_id
    )

    if meetings.empty:

        st.info(
            "No meetings found."
        )

    else:

        search = st.text_input(
            "🔎 Search meetings"
        )

        if search:

            filtered = meetings[
                meetings["filename"]
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        else:

            filtered = meetings


        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )


        st.divider()

        meeting_id = st.number_input(
            "Enter Meeting ID",
            min_value=1,
            step=1
        )


        col1, col2 = st.columns(2)


        # -----------------------------------------------------
        # VIEW
        # -----------------------------------------------------

        with col1:

            if st.button(
                "🔍 View Meeting",
                use_container_width=True
            ):

                meeting = get_meeting(
                    meeting_id,
                    st.session_state.user_id
                )

                if meeting:

                    (
                        mid,
                        uid,
                        filename,
                        transcript,
                        summary,
                        key_points,
                        action_items,
                        created_at
                    ) = meeting


                    st.subheader(
                        filename
                    )

                    st.caption(
                        f"Created: {created_at}"
                    )


                    st.write(
                        "### 📋 Summary"
                    )

                    st.info(
                        summary
                    )


                    st.write(
                        "### 💡 Key Points"
                    )

                    try:

                        points = json.loads(
                            key_points
                        )

                    except:

                        points = []


                    for point in points:

                        st.write(
                            f"✅ {point}"
                        )


                    st.write(
                        "### 📝 Transcript"
                    )

                    with st.expander(
                        "View transcript"
                    ):

                        st.write(
                            transcript
                        )


                    try:

                        tasks = json.loads(
                            action_items
                        )

                    except:

                        tasks = []


                    st.write(
                        "### ✅ Action Items"
                    )

                    if tasks:

                        task_table = []

                        for task in tasks:

                            task_table.append(

                                {
                                    "Task":
                                        task.get(
                                            "task",
                                            "Not specified"
                                        ),

                                    "Assigned To":
                                        task.get(
                                            "assigned_to",
                                            "Not specified"
                                        ),

                                    "Deadline":
                                        task.get(
                                            "deadline",
                                            "Not specified"
                                        ),

                                    "Priority":
                                        task.get(
                                            "priority",
                                            "Not specified"
                                        ),

                                    "Status":
                                        task.get(
                                            "status",
                                            "Pending"
                                        )
                                }

                            )

                        st.dataframe(
                            pd.DataFrame(
                                task_table
                            ),
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.info(
                            "No action items."
                        )


                else:

                    st.error(
                        "Meeting not found."
                    )


        # -----------------------------------------------------
        # DELETE
        # -----------------------------------------------------

        with col2:

            if st.button(
                "🗑️ Delete Meeting",
                use_container_width=True
            ):

                meeting = get_meeting(
                    meeting_id,
                    st.session_state.user_id
                )

                if meeting:

                    delete_meeting(
                        meeting_id,
                        st.session_state.user_id
                    )

                    st.success(
                        "Meeting deleted."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Meeting not found."
                    )


# =========================================================
# TASKS
# =========================================================

elif page == "✅ Tasks":

    st.title(
        "✅ Action Items"
    )

    meetings = get_user_meetings(
        st.session_state.user_id
    )

    all_tasks = []

    for _, row in meetings.iterrows():

        meeting_id = int(
            row["id"]
        )

        meeting = get_meeting(
            meeting_id,
            st.session_state.user_id
        )

        if not meeting:
            continue

        try:

            tasks = json.loads(
                meeting[6]
            )

        except:

            tasks = []


        for index, task in enumerate(tasks):

            all_tasks.append(

                {
                    "meeting_id":
                        meeting_id,

                    "task_index":
                        index,

                    "Meeting":
                        meeting[2],

                    "Task":
                        task.get(
                            "task",
                            "Not specified"
                        ),

                    "Assigned To":
                        task.get(
                            "assigned_to",
                            "Not specified"
                        ),

                    "Deadline":
                        task.get(
                            "deadline",
                            "Not specified"
                        ),

                    "Priority":
                        task.get(
                            "priority",
                            "Not specified"
                        ),

                    "Status":
                        task.get(
                            "status",
                            "Pending"
                        )
                }

            )


    if all_tasks:

        dataframe = pd.DataFrame(
            all_tasks
        )


        # -----------------------------------------------------
        # STATISTICS
        # -----------------------------------------------------

        total = len(dataframe)

        pending = len(
            dataframe[
                dataframe["Status"]
                != "Completed"
            ]
        )

        completed = len(
            dataframe[
                dataframe["Status"]
                == "Completed"
            ]
        )


        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Tasks",
            total
        )

        col2.metric(
            "⏳ Pending",
            pending
        )

        col3.metric(
            "✅ Completed",
            completed
        )


        st.divider()


        # -----------------------------------------------------
        # FILTER
        # -----------------------------------------------------

        status_filter = st.selectbox(
            "Filter tasks",
            [
                "All",
                "Pending",
                "Completed"
            ]
        )


        if status_filter != "All":

            display_df = dataframe[
                dataframe["Status"]
                == status_filter
            ]

        else:

            display_df = dataframe


        st.dataframe(
            display_df[
                [
                    "meeting_id",
                    "Meeting",
                    "Task",
                    "Assigned To",
                    "Deadline",
                    "Priority",
                    "Status"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        # -----------------------------------------------------
        # UPDATE STATUS
        # -----------------------------------------------------

        st.divider()

        st.subheader(
            "🔄 Update Task Status"
        )

        selected_meeting = st.number_input(
            "Meeting ID",
            min_value=1,
            step=1,
            key="task_meeting_id"
        )

        selected_task = st.number_input(
            "Task Number",
            min_value=1,
            step=1,
            key="task_number"
        )

        new_status = st.selectbox(
            "New Status",
            [
                "Pending",
                "Completed"
            ]
        )


        if st.button(
            "🔄 Update Status",
            use_container_width=True
        ):

            success = update_task_status(

                selected_meeting,

                st.session_state.user_id,

                selected_task - 1,

                new_status

            )

            if success:

                st.success(
                    "Task status updated."
                )

                st.rerun()

            else:

                st.error(
                    "Meeting or task not found."
                )


    else:

        st.info(
            "No action items available."
        )


# =========================================================
# SYSTEM STATUS
# =========================================================

elif page == "⚙️ System Status":

    st.title(
        "⚙️ System Status"
    )

    st.subheader(
        "🧠 AI Components"
    )


    # WHISPER

    st.write(
        "🎤 **Faster-Whisper**"
    )

    st.success(
        "Installed and configured for local transcription."
    )


    # OLLAMA

    st.write(
        "🤖 **Ollama**"
    )

    ollama_ok, message = check_ollama()

    if ollama_ok:

        st.success(
            message
        )

    else:

        st.error(
            message
        )


    # DATABASE

    st.write(
        "🗄️ **SQLite Database**"
    )

    if Path(DATABASE).exists():

        st.success(
            f"{DATABASE} is available."
        )

    else:

        st.warning(
            f"{DATABASE} does not exist yet."
        )


    st.divider()

    st.subheader(
        "📦 Configuration"
    )

    st.code(
        f"""
Ollama URL:
{OLLAMA_URL}

Ollama Model:
{OLLAMA_MODEL}

Database:
{DATABASE}

Whisper:
Faster-Whisper base

AI Mode:
100% Local
"""
    )


    st.info(
        "This project does not require an OpenAI API key. "
        "Whisper and Llama run locally."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Voice2Action • Stage 5 • "
    "Local AI Meeting Assistant • "
    "Faster-Whisper + Ollama + SQLite"
)