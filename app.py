# =========================================================
# VOICE2ACTION
# AI Meeting Assistant
# Gemini Cloud AI + SQLite
# =========================================================

import os
import json
import hashlib
import sqlite3
import tempfile

from pathlib import Path
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from google import genai

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

DATABASE = "voice2action.db"

# Gemini model
GEMINI_MODEL = "gemini-3.8-flash"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Voice2Action",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# GEMINI API KEY
# =========================================================

def get_gemini_api_key():

    try:
        return st.secrets["GEMINI_API_KEY"]

    except Exception:
        return os.getenv("GEMINI_API_KEY")


# =========================================================
# GEMINI AUDIO ANALYSIS
# =========================================================

def analyze_audio_with_gemini(audio_path):

    api_key = get_gemini_api_key()

    if not api_key:

        return (
            None,
            "Gemini API key is not configured."
        )

    try:

        # Create Gemini client
        client = genai.Client(
            api_key=api_key
        )

        # Upload audio to Gemini
        uploaded_file = client.files.upload(
            file=audio_path
        )

        # Prompt
        prompt = """
You are Voice2Action, a professional AI meeting assistant.

Analyze the uploaded meeting audio carefully.

Return ONLY valid JSON.

Use exactly this structure:

{
    "transcript": "Complete transcript of the meeting",

    "summary": "Short and clear summary of the meeting",

    "key_points": [
        "Important point 1",
        "Important point 2"
    ],

    "action_items": [
        {
            "task": "Task description",
            "assigned_to": "Person responsible or Not specified",
            "deadline": "Deadline or Not specified",
            "priority": "High, Medium, Low, or Not specified",
            "status": "Pending"
        }
    ]
}

Rules:

1. Accurately transcribe the meeting audio.
2. Do not invent information.
3. Include important discussions in the summary.
4. Extract the most important points.
5. Identify tasks or responsibilities mentioned in the meeting.
6. If the task owner is not mentioned, use "Not specified".
7. If the deadline is not mentioned, use "Not specified".
8. If priority is not mentioned, use "Not specified".
9. Every new action item must have status "Pending".
10. Return valid JSON only.
11. Do not include markdown.
12. Do not include explanations outside the JSON.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                uploaded_file,
                prompt
            ]
        )

        return response.text, None

    except Exception as e:

        return (
            None,
            str(e)
        )


# =========================================================
# DATABASE
# =========================================================

def get_connection():

    return sqlite3.connect(
        DATABASE
    )


def init_database():

    connection = get_connection()
    cursor = connection.cursor()

    # USERS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    # MEETINGS TABLE
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


# Initialize database
init_database()


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()


# =========================================================
# REGISTER USER
# =========================================================

def register_user(
    username,
    password
):

    connection = get_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                password
            )
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

def login_user(
    username,
    password
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            username
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
# GET SINGLE MEETING
# =========================================================

def get_meeting(
    meeting_id,
    user_id
):

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

def delete_meeting(
    meeting_id,
    user_id
):

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

    try:

        tasks = json.loads(
            meeting[6]
        )

    except (
        json.JSONDecodeError,
        TypeError
    ):

        return False

    if (
        task_index < 0
        or task_index >= len(tasks)
    ):

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
# PARSE AI JSON
# =========================================================

def parse_ai_response(response):

    if not response:

        return None

    response = response.strip()

    # -----------------------------------------------------
    # Remove markdown code fences
    # -----------------------------------------------------

    if response.startswith("```"):

        lines = response.splitlines()

        if lines:

            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):

            lines = lines[:-1]

        response = "\n".join(
            lines
        ).strip()

    # -----------------------------------------------------
    # Direct JSON
    # -----------------------------------------------------

    try:

        return json.loads(
            response
        )

    except json.JSONDecodeError:

        pass

    # -----------------------------------------------------
    # Find JSON object inside response
    # -----------------------------------------------------

    start = response.find("{")
    end = response.rfind("}")

    if (
        start != -1
        and end != -1
    ):

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
# NORMALIZE AI ANALYSIS
# =========================================================

def normalize_analysis(
    analysis
):

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
                "task": str(
                    task.get(
                        "task",
                        "Not specified"
                    )
                ),

                "assigned_to": str(
                    task.get(
                        "assigned_to",
                        "Not specified"
                    )
                ),

                "deadline": str(
                    task.get(
                        "deadline",
                        "Not specified"
                    )
                ),

                "priority": str(
                    task.get(
                        "priority",
                        "Not specified"
                    )
                ),

                "status": str(
                    task.get(
                        "status",
                        "Pending"
                    )
                )
            }
        )

    return {
        "summary": str(summary),

        "key_points": [
            str(point)
            for point in key_points
        ],

        "action_items":
            normalized_tasks
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

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    elements.append(
        Paragraph(
            "Voice2Action Meeting Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(
            1,
            20
        )
    )

    # -----------------------------------------------------
    # MEETING
    # -----------------------------------------------------

    elements.append(
        Paragraph(
            f"Meeting: {filename}",
            styles["Heading2"]
        )
    )

    elements.append(
        Spacer(
            1,
            10
        )
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    elements.append(
        Paragraph(
            "Summary",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            str(summary),
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(
            1,
            15
        )
    )

    # -----------------------------------------------------
    # KEY POINTS
    # -----------------------------------------------------

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
        Spacer(
            1,
            15
        )
    )

    # -----------------------------------------------------
    # ACTION ITEMS
    # -----------------------------------------------------

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

    if len(table_data) == 1:

        table_data.append(
            [
                "No action items",
                "-",
                "-",
                "-",
                "-"
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

    elements.append(
        table
    )

    document.build(
        elements
    )

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

    st.title(
        "🎙️ Voice2Action"
    )

    st.subheader(
        "AI Meeting Assistant"
    )

    st.info(
        "☁️ Powered by Gemini Cloud AI"
    )

    # -----------------------------------------------------
    # LOGIN / REGISTER TABS
    # -----------------------------------------------------

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Register"
        ]
    )

    # =====================================================
    # LOGIN
    # =====================================================

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

                st.session_state.user_id = (
                    user[0]
                )

                st.session_state.username = (
                    user[1]
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    # =====================================================
    # REGISTER
    # =====================================================

    with register_tab:

        new_username = st.text_input(
            "Create Username",
            key="register_username"
        )

        new_password = st.text_input(
            "Create Password",
            type="password",
            key="register_password"
        )

        if st.button(
            "📝 Create Account",
            use_container_width=True
        ):

            if (
                not new_username
                or not new_password
            ):

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

    st.header(
        "🎙️ Voice2Action"
    )

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

        st.session_state.last_filename = None

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

    st.title(
        "📊 Dashboard"
    )

    st.write(
        f"Welcome, **{st.session_state.username}** 👋"
    )

    # -----------------------------------------------------
    # MEETING STATISTICS
    # -----------------------------------------------------

    total_meetings = len(
        meetings
    )

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

            except (
                json.JSONDecodeError,
                TypeError
            ):

                tasks = []

            total_tasks += len(
                tasks
            )

            for task in tasks:

                if (
                    task.get("status")
                    == "Completed"
                ):

                    completed_tasks += 1

                else:

                    pending_tasks += 1

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # GEMINI STATUS
    # -----------------------------------------------------

    st.subheader(
        "☁️ Gemini Cloud AI"
    )

    if get_gemini_api_key():

        st.success(
            f"🟢 Gemini is configured "
            f"using {GEMINI_MODEL}"
        )

    else:

        st.error(
            "🔴 Gemini API key is not configured."
        )

        st.caption(
            "Add GEMINI_API_KEY to Streamlit Secrets."
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
        "structured information using Gemini AI."
    )

    st.info(
        "Supported formats: WAV, MP3, M4A, FLAC, OGG"
    )

    # -----------------------------------------------------
    # AUDIO UPLOAD
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload meeting audio",
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

        # -------------------------------------------------
        # GEMINI STATUS
        # -------------------------------------------------

        st.info(
            "☁️ Powered by Gemini Cloud AI"
        )

        if not get_gemini_api_key():

            st.warning(
                "Gemini API key is not configured. "
                "Please add GEMINI_API_KEY in Streamlit Secrets."
            )

        # -------------------------------------------------
        # MEETING TITLE
        # -------------------------------------------------

        meeting_title = st.text_input(
            "Meeting title",
            value=Path(
                uploaded_file.name
            ).stem,
            key="meeting_title"
        )

        # -------------------------------------------------
        # ANALYZE BUTTON
        # -------------------------------------------------

        if st.button(
            "🚀 Analyze Meeting",
            type="primary",
            use_container_width=True
        ):

            if not get_gemini_api_key():

                st.error(
                    "Gemini API key is not configured."
                )

                st.stop()

            audio_path = None

            try:

                # -----------------------------------------
                # SAVE TEMPORARY AUDIO
                # -----------------------------------------

                extension = Path(
                    uploaded_file.name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temp:

                    temp.write(
                        uploaded_file.getvalue()
                    )

                    audio_path = temp.name

                # -----------------------------------------
                # GEMINI ANALYSIS
                # -----------------------------------------

                with st.spinner(
                    "☁️ Gemini AI is analyzing your meeting..."
                ):

                    response, error = (
                        analyze_audio_with_gemini(
                            audio_path
                        )
                    )

                # -----------------------------------------
                # AI ERROR
                # -----------------------------------------

                if error:

                    st.error(
                        f"AI analysis failed: {error}"
                    )

                    st.stop()

                # -----------------------------------------
                # PARSE AI RESPONSE
                # -----------------------------------------

                parsed = parse_ai_response(
                    response
                )

                if not parsed:

                    st.error(
                        "Gemini returned invalid JSON."
                    )

                    with st.expander(
                        "Show Gemini response"
                    ):

                        st.code(
                            response
                            or "No response"
                        )

                    st.stop()

                # -----------------------------------------
                # EXTRACT TRANSCRIPT
                # -----------------------------------------

                transcript = parsed.get(
                    "transcript",
                    ""
                )

                if not transcript:

                    transcript = (
                        "Transcript not available."
                    )

                # -----------------------------------------
                # NORMALIZE ANALYSIS
                # -----------------------------------------

                analysis = normalize_analysis(
                    parsed
                )

                if not analysis:

                    st.error(
                        "Unable to process Gemini analysis."
                    )

                    st.stop()

                # -----------------------------------------
                # SAVE SESSION
                # -----------------------------------------

                st.session_state.analysis = (
                    analysis
                )

                st.session_state.last_transcript = (
                    transcript
                )

                st.session_state.last_filename = (
                    meeting_title
                )

                # -----------------------------------------
                # SAVE DATABASE
                # -----------------------------------------

                save_meeting(
                    st.session_state.user_id,
                    meeting_title,
                    transcript,
                    analysis["summary"],
                    analysis["key_points"],
                    analysis["action_items"]
                )

                st.success(
                    "💾 Meeting analyzed and saved successfully."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )

            finally:

                # -----------------------------------------
                # REMOVE TEMP AUDIO FILE
                # -----------------------------------------

                if audio_path:

                    try:

                        os.remove(
                            audio_path
                        )

                    except OSError:

                        pass

    # =====================================================
    # DISPLAY LAST ANALYSIS
    # =====================================================

    if (
        st.session_state.analysis
        and st.session_state.last_transcript
    ):

        analysis = (
            st.session_state.analysis
        )

        transcript = (
            st.session_state.last_transcript
        )

        filename = (
            st.session_state.last_filename
        )

        st.divider()

        # -------------------------------------------------
        # TRANSCRIPT
        # -------------------------------------------------

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

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        st.header(
            "📋 Summary"
        )

        st.info(
            analysis["summary"]
        )

        # -------------------------------------------------
        # KEY POINTS
        # -------------------------------------------------

        st.header(
            "💡 Key Points"
        )

        if analysis["key_points"]:

            for point in analysis[
                "key_points"
            ]:

                st.write(
                    f"✅ {point}"
                )

        else:

            st.write(
                "No key points detected."
            )

        # -------------------------------------------------
        # ACTION ITEMS
        # -------------------------------------------------

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
                pd.DataFrame(
                    task_table
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No action items detected."
            )

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "📥 Export"
        )

        col1, col2, col3 = st.columns(3)

        # ---------------------------------------------
        # TRANSCRIPT DOWNLOAD
        # ---------------------------------------------

        with col1:

            st.download_button(
                "📝 Transcript",
                transcript,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

        # ---------------------------------------------
        # JSON DOWNLOAD
        # ---------------------------------------------

        with col2:

            json_data = json.dumps(
                {
                    "transcript":
                        transcript,

                    "summary":
                        analysis["summary"],

                    "key_points":
                        analysis["key_points"],

                    "action_items":
                        analysis["action_items"]
                },
                indent=4
            )

            st.download_button(
                "🧠 AI Analysis",
                json_data,
                file_name="meeting_analysis.json",
                mime="application/json",
                use_container_width=True
            )

        # ---------------------------------------------
        # PDF DOWNLOAD
        # ---------------------------------------------

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
                file_name="meeting_report.pdf",
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

        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------

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

        # -------------------------------------------------
        # MEETING ID
        # -------------------------------------------------

        meeting_id = st.number_input(
            "Enter Meeting ID",
            min_value=1,
            step=1,
            key="history_meeting_id"
        )

        col1, col2 = st.columns(2)

        # =================================================
        # VIEW MEETING
        # =================================================

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

                    # Summary
                    st.write(
                        "### 📋 Summary"
                    )

                    st.info(
                        summary
                    )

                    # Key points
                    st.write(
                        "### 💡 Key Points"
                    )

                    try:

                        points = json.loads(
                            key_points
                        )

                    except (
                        json.JSONDecodeError,
                        TypeError
                    ):

                        points = []

                    for point in points:

                        st.write(
                            f"✅ {point}"
                        )

                    # Transcript
                    st.write(
                        "### 📝 Transcript"
                    )

                    with st.expander(
                        "View transcript"
                    ):

                        st.write(
                            transcript
                        )

                    # Action items
                    st.write(
                        "### ✅ Action Items"
                    )

                    try:

                        tasks = json.loads(
                            action_items
                        )

                    except (
                        json.JSONDecodeError,
                        TypeError
                    ):

                        tasks = []

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

        # =================================================
        # DELETE MEETING
        # =================================================

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

    # -----------------------------------------------------
    # COLLECT ALL TASKS
    # -----------------------------------------------------

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

        except (
            json.JSONDecodeError,
            TypeError
        ):

            tasks = []

        for index, task in enumerate(
            tasks
        ):

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

    # =====================================================
    # DISPLAY TASKS
    # =====================================================

    if all_tasks:

        dataframe = pd.DataFrame(
            all_tasks
        )

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        total = len(
            dataframe
        )

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

        # -------------------------------------------------
        # FILTER
        # -------------------------------------------------

        status_filter = st.selectbox(
            "Filter tasks",
            [
                "All",
                "Pending",
                "Completed"
            ],
            key="task_filter"
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

        # -------------------------------------------------
        # UPDATE TASK STATUS
        # -------------------------------------------------

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
            ],
            key="new_task_status"
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

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    st.write(
        "☁️ **Gemini Cloud AI**"
    )

    if get_gemini_api_key():

        st.success(
            f"Gemini is configured successfully "
            f"using {GEMINI_MODEL}."
        )

    else:

        st.error(
            "Gemini API key is not configured."
        )

        st.info(
            "Add GEMINI_API_KEY in Streamlit Cloud Secrets."
        )

    # -----------------------------------------------------
    # AUDIO PROCESSING
    # -----------------------------------------------------

    st.write(
        "🎵 **Audio Processing**"
    )

    st.success(
        "Audio is uploaded directly to Gemini for analysis."
    )

    # -----------------------------------------------------
    # DATABASE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "📦 Configuration"
    )

    gemini_status = (
        "Configured"
        if get_gemini_api_key()
        else "Not configured"
    )

    st.code(
        f"""
AI Provider:
Gemini Cloud AI

Gemini Model:
{GEMINI_MODEL}

Gemini API:
{gemini_status}

Database:
{DATABASE}

Transcription:
Gemini Audio Understanding

AI Analysis:
Gemini Cloud AI

Local AI:
Disabled

Ollama:
Removed

OpenAI:
Removed
"""
    )

    st.info(
        "Voice2Action uses Gemini Cloud AI to "
        "transcribe and analyze meeting recordings."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Voice2Action • Stage 5 • "
    "AI Meeting Assistant • "
    "Gemini Cloud AI + SQLite"
)