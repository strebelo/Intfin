"""Classroom 2/3-of-the-average guessing game for Streamlit.

Run locally with: streamlit run app.py
Configure the instructor passphrase in .streamlit/secrets.toml.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="2/3 of the Average", page_icon="🎯", layout="centered")

TITLE = os.getenv("GAME_TITLE", "2/3 of the Average")
LOWER_BOUND = int(os.getenv("NUMBER_MIN", "0"))
UPPER_BOUND = int(os.getenv("NUMBER_MAX", "100"))
DATA_DIR = Path(os.getenv("STREAMLIT_DATA_DIR", Path(__file__).resolve().parent))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "classroom_game.sqlite3"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=20000")
    return conn


def initialize() -> None:
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS game_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            revealed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            number INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.execute(
            "INSERT OR IGNORE INTO game_state (id, revealed, updated_at) VALUES (1, 0, ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )


def get_game_state() -> bool:
    with connect() as conn:
        row = conn.execute("SELECT revealed FROM game_state WHERE id = 1").fetchone()
    return bool(row["revealed"])


def get_submissions() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute("SELECT name, number FROM submissions ORDER BY name COLLATE NOCASE").fetchall()
    return pd.DataFrame([dict(row) for row in rows], columns=["name", "number"])


def save_submission(name: str, number: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO submissions (name, number, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET number=excluded.number, updated_at=excluded.updated_at""",
            (name.strip(), int(number), now),
        )


def reveal_results() -> None:
    with connect() as conn:
        conn.execute("UPDATE game_state SET revealed = 1, updated_at = ? WHERE id = 1",
                     (datetime.now(timezone.utc).isoformat(),))


def reset_game() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM submissions")
        conn.execute("UPDATE game_state SET revealed = 0, updated_at = ? WHERE id = 1",
                     (datetime.now(timezone.utc).isoformat(),))


def get_admin_password() -> str | None:
    try:
        return st.secrets["instructor_password"]
    except Exception:
        return os.getenv("INSTRUCTOR_PASSWORD")


initialize()
st.title(f"🎯 {TITLE}")
st.caption(f"Choose an integer from {LOWER_BOUND} to {UPPER_BOUND}. The winning guess is closest to two-thirds of the class average.")

# The instructor unlocks controls with a private passphrase; students never see the controls by default.
with st.sidebar:
    st.subheader("Instructor")
    password = get_admin_password()
    if not password:
        st.warning("Set `instructor_password` in Streamlit secrets before sharing this app.")
        is_admin = False
    else:
        entered = st.text_input("Instructor passphrase", type="password", key="admin_passphrase")
        is_admin = bool(entered) and entered == password
        if entered and not is_admin:
            st.caption("Passphrase not recognized.")

revealed = get_game_state()
submissions = get_submissions()

if not revealed:
    st.subheader("Submit your guess")
    with st.form("student_submission", clear_on_submit=False):
        name = st.text_input("Your name", max_chars=80, placeholder="First and last name")
        number = st.number_input("Your number", min_value=LOWER_BOUND, max_value=UPPER_BOUND, value=50, step=1)
        submitted = st.form_submit_button("Submit or update my guess", type="primary", use_container_width=True)
    if submitted:
        clean_name = name.strip()
        if not clean_name:
            st.error("Please enter your name.")
        elif len(clean_name) > 80:
            st.error("Please use a name with 80 characters or fewer.")
        else:
            save_submission(clean_name, int(number))
            st.success("Your guess has been recorded. You can return to this page to update it before the instructor reveals the results.")
    st.info("Your guess and the class list will stay hidden until the instructor reveals the results.")
    st.caption("If two students enter the same name, the later guess under that name replaces the earlier one. Use distinct names (for example, include a last initial) if needed.")
else:
    st.subheader("Class results")
    if submissions.empty:
        st.warning("No guesses were submitted.")
    else:
        mean = float(submissions["number"].mean())
        target = (2 / 3) * mean
        distances = (submissions["number"] - target).abs()
        smallest_gap = float(distances.min())
        winners = submissions.loc[distances == smallest_gap, "name"].tolist()
        results = submissions.assign(
            **{"Absolute distance from target": distances}
        ).rename(columns={"name": "Student", "number": "Number"})
        results["Absolute distance from target"] = results["Absolute distance from target"].map(lambda x: f"{x:.2f}")
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.metric("Class average", f"{mean:.2f}")
        st.metric("Two-thirds of the average", f"{target:.2f}")
        if len(winners) == 1:
            st.success(f"**{winners[0]}** is closest to the target, with a guess of {int(submissions.loc[submissions['name'] == winners[0], 'number'].iloc[0])}.")
        else:
            st.success("Tie: " + ", ".join(f"**{winner}**" for winner in winners) + f" are equally close (distance {smallest_gap:.2f}).")

if is_admin:
    st.divider()
    st.subheader("Instructor controls")
    latest = get_submissions()
    st.write(f"**{len(latest)} submission(s)** recorded")
    if latest.empty:
        st.caption("The student list will appear here as guesses arrive.")
    else:
        st.dataframe(latest.rename(columns={"name": "Student", "number": "Number"}), hide_index=True, use_container_width=True)
    if not get_game_state():
        if st.button("Confirm submissions and reveal results", type="primary", use_container_width=True):
            reveal_results()
            st.rerun()
    else:
        st.success("Results are revealed and submissions are locked.")
    with st.expander("Start a new round"):
        st.warning("This permanently clears the current round's submissions and results.")
        if st.button("Clear submissions and reopen", type="secondary"):
            reset_game()
            st.rerun()
