"""Conversation Session Manager for Multi-Turn Regulation Chats"""
import os
import json
import time
import uuid
from typing import List, Dict, Optional, Any
from app.models import ChatSession, ChatMessage

SESSIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "sessions.json")

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.load_sessions()

    def create_session(self, title: Optional[str] = "New Conversation") -> ChatSession:
        sid = f"sess_{uuid.uuid4().hex[:10]}"
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        session = ChatSession(
            id=sid,
            title=title or "New Regulation Chat",
            created_at=now,
            updated_at=now,
            messages=[]
        )
        self.sessions[sid] = session
        self.save_sessions()
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[ChatSession]:
        sorted_list = sorted(self.sessions.values(), key=lambda s: s.updated_at, reverse=True)
        return sorted_list

    def add_message(self, session_id: str, message: ChatMessage):
        session = self.get_session(session_id)
        if not session:
            session = self.create_session()
            session_id = session.id
        
        session.messages.append(message)
        session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Auto-update title from first user message
        if len(session.messages) == 1 or session.title.startswith("New"):
            first_user = next((m for m in session.messages if m.role == "user"), None)
            if first_user:
                session.title = first_user.content[:40] + ("..." if len(first_user.content) > 40 else "")

        self.save_sessions()
        return session_id

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions()
            return True
        return False

    def update_feedback(self, session_id: str, msg_idx: int, feedback: str):
        session = self.get_session(session_id)
        if session and 0 <= msg_idx < len(session.messages):
            session.messages[msg_idx].feedback = feedback
            self.save_sessions()
            return True
        return False

    def load_sessions(self):
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, sdata in data.items():
                        self.sessions[sid] = ChatSession(**sdata)
            except Exception as e:
                print(f"[SessionManager] Error loading sessions: {e}")

    def save_sessions(self):
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        try:
            with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
                serializable = {sid: s.model_dump() for sid, s in self.sessions.items()}
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"[SessionManager] Error saving sessions: {e}")
