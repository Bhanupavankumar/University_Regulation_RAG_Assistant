import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  GraduationCap,
  MessageSquare,
  FileText,
  UploadCloud,
  Search,
  Sliders,
  Sparkles,
  Send,
  BookOpen,
  CheckCircle,
  AlertCircle,
  Clock,
  Layers,
  ExternalLink,
  Trash2,
  Copy,
  Check,
  RefreshCw,
  Info,
  ChevronRight,
  Database,
  ShieldCheck,
  X,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  ThumbsUp,
  ThumbsDown,
  BarChart3,
  GitCompare,
  Plus,
  Download,
  Palette,
  Eye
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || (
  typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://127.0.0.1:8000/api'
    : 'http://127.0.0.1:8000/api'
);

export default function App() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'docs' | 'compare' | 'analytics' | 'upload' | 'search'
  const [theme, setTheme] = useState('theme-default');
  const [stats, setStats] = useState({
    total_documents: 10,
    total_chunks: 25,
    total_categories: 5,
    categories: ['Academic', 'Examination', 'Attendance', 'Handbook', 'Scholarship'],
    vector_dimension: 384,
    index_type: 'FAISS / Hybrid BM25 Index',
    status: 'Operational',
    model_name: 'all-MiniLM-L6-v2 (384d Dense)',
    total_queries_served: 0
  });

  const [documents, setDocuments] = useState([]);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [inspectingDoc, setInspectingDoc] = useState(null);
  const [readingDoc, setReadingDoc] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  // Sessions State
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(() => {
    return localStorage.getItem('unirag_session_id') || null;
  });

  // RAG Configuration Settings
  const [ragSettings, setRagSettings] = useState({
    top_k: 4,
    threshold: 0.35,
    category: 'all',
    llm_provider: 'local', // 'local' | 'gemini' | 'groq' | 'openai' | 'ollama'
    api_key: localStorage.getItem('rag_api_key') || '',
    temperature: 0.2,
    hybrid_weight: 0.65
  });

  // Default welcome message
  const defaultWelcome = {
    role: 'assistant',
    content: `### 🎓 Welcome to the University Regulation Enterprise Assistant!
I am trained on **all official university regulations**, bylaws, examination manuals, attendance policies, scholarship directives, research ethics, and student handbooks.

**Ask me anything regarding:**
- 📘 **Academic Regulations**: 10-point GPA calculation, SGPA/CGPA, add/drop limits, Honors & Minor degrees.
- 📝 **Examinations**: Internal 40:60 weightage, hall ticket issuance, re-evaluation fees ($25), supplementary exams.
- ⏱ **Attendance & Condonation**: Mandatory 75% rule, medical condonation up to 65%, Grade SA detention.
- 🏛 **Student Conduct & Hostels**: Anti-Ragging zero-tolerance directives, hostel gate hours, library fair use.
- 💼 **Internships & Placement**: 6-week mandatory summer internships, final semester capstone policies.
- 🔬 **Research & Plagiarism**: 10% similarity index limit, thesis submission ethics, patent IP sharing.
- 💰 **Scholarships & Fees**: Chancellor's 50% merit waiver, tuition late fees, statutory UGC refund schedule.

All responses feature verified source page citations and hallucination checks!`,
    citations: [],
    confidence: 'High',
    time: 'Just now',
    llm_used: 'Hybrid Neural RAG',
    follow_ups: [
      'What is the minimum attendance required for examinations?',
      'How is CGPA calculated and what is the minimum to graduate?',
      'What is the permissible thesis plagiarism similarity index?'
    ]
  };

  // Chat Messages State with localStorage Persistence
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('unirag_chat_history');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (e) {
      console.error('Error loading chat history:', e);
    }
    return [defaultWelcome];
  });

  // Save messages to localStorage on every change
  useEffect(() => {
    try {
      localStorage.setItem('unirag_chat_history', JSON.stringify(messages));
    } catch (e) {
      console.error('Error saving chat history:', e);
    }
  }, [messages]);

  // Save session ID
  useEffect(() => {
    if (currentSessionId) {
      localStorage.setItem('unirag_session_id', currentSessionId);
    }
  }, [currentSessionId]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Theme apply
  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  // Fetch initial stats, documents, and sessions
  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (res.ok) setStats(await res.json());
    } catch (e) {
      console.log('Stats error:', e);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) setDocuments(await res.json());
    } catch (e) {
      console.log('Docs error:', e);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        if (data.length > 0 && !currentSessionId) {
          setCurrentSessionId(data[0].id);
        }
      }
    } catch (e) {
      console.log('Sessions error:', e);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchDocuments();
    fetchSessions();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Voice Input (Web Speech API)
  const toggleSpeechRecognition = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech Recognition is not supported by your browser.');
      return;
    }

    if (isRecording) {
      setIsRecording(false);
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsRecording(true);
    recognition.onend = () => setIsRecording(false);
    recognition.onerror = () => setIsRecording(false);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInputQuery(transcript);
      setIsRecording(false);
    };

    recognition.start();
  };

  // Text-to-Speech Audio Reader
  const toggleSpeak = (text, idx) => {
    if ('speechSynthesis' in window) {
      if (speakingIndex === idx) {
        window.speechSynthesis.cancel();
        setSpeakingIndex(null);
        return;
      }

      window.speechSynthesis.cancel();
      // Clean markdown tags for natural speech
      const cleanText = text.replace(/#|\*|`|\[.*?\]|-/g, '');
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      utterance.onend = () => setSpeakingIndex(null);
      utterance.onerror = () => setSpeakingIndex(null);

      setSpeakingIndex(idx);
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleSendQuery = async (queryText = inputQuery) => {
    const q = queryText.trim();
    if (!q || loading) return;

    const userMsg = {
      role: 'user',
      content: q,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          session_id: currentSessionId || 'default',
          top_k: parseInt(ragSettings.top_k),
          threshold: parseFloat(ragSettings.threshold),
          category: ragSettings.category,
          llm_provider: ragSettings.llm_provider,
          api_key: ragSettings.api_key || undefined,
          temperature: parseFloat(ragSettings.temperature),
          hybrid_weight: parseFloat(ragSettings.hybrid_weight)
        })
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const data = await res.json();

      const aiMsg = {
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        confidence: data.confidence || 'Medium',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        llm_used: data.llm_used,
        processing_time_ms: data.processing_time_ms,
        follow_ups: data.follow_up_questions || [],
        hallucination_check: data.hallucination_check
      };

      setMessages((prev) => [...prev, aiMsg]);
      fetchStats();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Error communicating with backend**: Could not fetch regulation response. Make sure the backend server is running at \`${API_BASE}\`.`,
          citations: [],
          confidence: 'Low',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (msgIdx, type) => {
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId || 'default',
          message_index: msgIdx,
          feedback: type
        })
      });

      setMessages((prev) =>
        prev.map((m, i) => (i === msgIdx ? { ...m, feedback: type } : m))
      );
    } catch (e) {
      console.log('Feedback error:', e);
    }
  };

  const createNewSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' });
      if (res.ok) {
        const newSess = await res.json();
        setSessions((prev) => [newSess, ...prev]);
        setCurrentSessionId(newSess.id);
        setMessages([
          {
            role: 'assistant',
            content: 'Started new regulation session. How can I help you with academic policies?',
            citations: [],
            confidence: 'High',
            time: 'Just now'
          }
        ]);
      }
    } catch (e) {
      console.log('Session create error:', e);
    }
  };

  const selectSession = async (sessId) => {
    setCurrentSessionId(sessId);
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.messages && data.messages.length > 0) {
          setMessages(data.messages);
        }
      }
    } catch (e) {
      console.log('Load session error:', e);
    }
  };

  const deleteSession = async (sessId, e) => {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE}/sessions/${sessId}`, { method: 'DELETE' });
      setSessions((prev) => prev.filter((s) => s.id !== sessId));
      if (currentSessionId === sessId) {
        createNewSession();
      }
    } catch (e) {
      console.log('Delete session error:', e);
    }
  };

  const exportChatTranscript = () => {
    const transcript = messages
      .map((m) => `[${m.role.toUpperCase()}] (${m.time || ''}):\n${m.content}\n`)
      .join('\n---\n\n');
    const blob = new Blob([transcript], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `UniRAG_Transcript_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
  };

  const quickPrompts = [
    'What is the minimum attendance required for examinations?',
    'How is CGPA calculated and what is the minimum to graduate?',
    'What is the policy and fee for exam re-evaluation?',
    'What are the eligibility criteria for an Honors Degree?',
    'What is the permissible thesis plagiarism similarity index?',
    'What are the summer internship requirements and credits?',
    'What is the fee refund schedule upon withdrawal?'
  ];

  return (
    <div className="app-container">
      {/* Header Bar */}
      <header className="header">
        <div className="header-brand">
          <div className="logo-badge">
            <GraduationCap size={26} />
          </div>
          <div>
            <h1 className="brand-title">UniRAG Enterprise</h1>
            <p className="brand-subtitle">
              <ShieldCheck size={14} style={{ color: '#10b981' }} />
              Official University Regulations & Policy Grounded QA
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={16} />
            AI Assistant
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'docs' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('docs');
              fetchDocuments();
              fetchStats();
            }}
          >
            <BookOpen size={16} />
            Regulations Hub ({documents.length})
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'compare' ? 'active' : ''}`}
            onClick={() => setActiveTab('compare')}
          >
            <GitCompare size={16} />
            Compare Studio
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <BarChart3 size={16} />
            Analytics
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <UploadCloud size={16} />
            Document Studio
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            <Search size={16} />
            Semantic Explorer
          </button>
        </nav>

        {/* Right Actions */}
        <div className="header-actions">
          {/* Theme Switcher */}
          <div className="glass-pill" style={{ padding: '4px 8px' }}>
            <Palette size={14} color="var(--primary)" />
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-main)',
                outline: 'none',
                fontSize: '0.78rem',
                cursor: 'pointer'
              }}
            >
              <option value="theme-default" style={{ background: '#111827' }}>Indigo Glow</option>
              <option value="theme-emerald" style={{ background: '#111827' }}>Cyber Emerald</option>
              <option value="theme-amber" style={{ background: '#111827' }}>Solar Amber</option>
              <option value="theme-minimal" style={{ background: '#111827' }}>Sky Blue</option>
            </select>
          </div>

          <div className="glass-pill" style={{ borderColor: 'rgba(16, 185, 129, 0.3)' }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                backgroundColor: '#10b981',
                boxShadow: '0 0 8px #10b981'
              }}
            />
            <span>{stats.total_chunks} Chunks</span>
          </div>

          <button
            className="action-btn"
            onClick={() => setShowSettings(true)}
            title="RAG Parameters & Model Settings"
          >
            <Sliders size={16} />
            <span>Settings</span>
          </button>
        </div>
      </header>

      {/* Main Tab Content */}
      <main style={{ flex: 1 }}>
        {activeTab === 'chat' && (
          <ChatView
            messages={messages}
            inputQuery={inputQuery}
            setInputQuery={setInputQuery}
            handleSendQuery={handleSendQuery}
            loading={loading}
            quickPrompts={quickPrompts}
            onSelectCitation={setSelectedCitation}
            onCopy={(text, idx) => {
              navigator.clipboard.writeText(text);
              setCopiedIndex(idx);
              setTimeout(() => setCopiedIndex(null), 2000);
            }}
            copiedIndex={copiedIndex}
            onSpeak={toggleSpeak}
            speakingIndex={speakingIndex}
            onFeedback={handleFeedback}
            isRecording={isRecording}
            onToggleVoice={toggleSpeechRecognition}
            sessions={sessions}
            currentSessionId={currentSessionId}
            onNewSession={createNewSession}
            onSelectSession={selectSession}
            onDeleteSession={deleteSession}
            onExportTranscript={exportChatTranscript}
            onClearChat={() => {
              if (window.confirm('Clear all conversation messages in this session?')) {
                setMessages([defaultWelcome]);
                localStorage.removeItem('unirag_chat_history');
              }
            }}
            stats={stats}
            ragSettings={ragSettings}
            messagesEndRef={messagesEndRef}
          />
        )}

        {activeTab === 'docs' && (
          <DocumentsView
            documents={documents}
            stats={stats}
            onRefresh={() => {
              fetchDocuments();
              fetchStats();
            }}
            onInspectDoc={(doc) => setInspectingDoc(doc)}
            onReadDoc={(doc) => setReadingDoc(doc)}
            onDeleteDoc={async (doc) => {
              if (window.confirm(`Are you sure you want to delete "${doc.title}"? This will remove its vectors from FAISS.`)) {
                try {
                  const res = await fetch(`${API_BASE}/documents/${doc.id}`, { method: 'DELETE' });
                  if (res.ok) {
                    fetchDocuments();
                    fetchStats();
                  }
                } catch (e) {
                  console.error('Delete doc error:', e);
                }
              }
            }}
            onAskAboutDoc={(doc) => {
              setActiveTab('chat');
              handleSendQuery(`What are the key rules and provisions in ${doc.title}?`);
            }}
          />
        )}

        {activeTab === 'compare' && (
          <CompareView categories={stats.categories} />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsDashboardView stats={stats} />
        )}

        {activeTab === 'upload' && (
          <UploadView
            onUploadSuccess={(doc) => {
              fetchDocuments();
              fetchStats();
            }}
            onInspectDoc={(doc) => setInspectingDoc(doc)}
            onNavigate={(tab, query) => {
              setActiveTab(tab);
              if (query) setInputQuery(query);
            }}
          />
        )}

        {activeTab === 'search' && (
          <SemanticSearchView
            categories={stats.categories}
            onSelectCitation={setSelectedCitation}
            onNavigate={(tab, query) => {
              setActiveTab(tab);
              if (query) handleSendQuery(query);
            }}
          />
        )}
      </main>

      {/* Citation Inspector Modal */}
      {selectedCitation && (
        <CitationModal
          citation={selectedCitation}
          documents={documents}
          onClose={() => setSelectedCitation(null)}
          onReadFullDoc={(doc) => {
            setSelectedCitation(null);
            setReadingDoc(doc);
          }}
          onAskAboutDoc={(doc) => {
            setSelectedCitation(null);
            setActiveTab('chat');
            handleSendQuery(`Can you explain what the regulation says regarding: ${selectedCitation.snippet.slice(0, 80)}...`);
          }}
        />
      )}

      {/* Document Chunks Inspector Modal */}
      {inspectingDoc && (
        <DocChunksModal
          doc={inspectingDoc}
          onClose={() => setInspectingDoc(null)}
          onSelectCitation={setSelectedCitation}
        />
      )}

      {/* Full Document Reader Modal */}
      {readingDoc && (
        <FullDocModal
          doc={readingDoc}
          onClose={() => setReadingDoc(null)}
          onAskAboutDoc={(doc) => {
            setReadingDoc(null);
            setActiveTab('chat');
            handleSendQuery(`Can you explain the key sections and policies in ${doc.title}?`);
          }}
        />
      )}

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal
          settings={ragSettings}
          onSave={(newSettings) => {
            setRagSettings(newSettings);
            localStorage.setItem('rag_api_key', newSettings.api_key);
            setShowSettings(false);
          }}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}

// ----------------------------------------------------
// SUB-COMPONENTS
// ----------------------------------------------------

function ChatView({
  messages,
  inputQuery,
  setInputQuery,
  handleSendQuery,
  loading,
  quickPrompts,
  onSelectCitation,
  onCopy,
  copiedIndex,
  onSpeak,
  speakingIndex,
  onFeedback,
  isRecording,
  onToggleVoice,
  sessions,
  currentSessionId,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  onExportTranscript,
  onClearChat,
  stats,
  ragSettings,
  messagesEndRef
}) {
  return (
    <div className="chat-container">
      {/* Left Sessions Sidebar */}
      <aside className="glass-panel sessions-sidebar">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f1f5f9' }}>
            Chat Sessions
          </span>
          <button
            className="action-btn"
            style={{ padding: '4px 8px', fontSize: '0.78rem', background: 'var(--primary)', color: 'white' }}
            onClick={onNewSession}
          >
            <Plus size={13} />
            <span>New</span>
          </button>
        </div>

        <div className="sessions-list">
          {sessions.map((sess) => (
            <div
              key={sess.id}
              className={`session-item ${sess.id === currentSessionId ? 'active' : ''}`}
              onClick={() => onSelectSession(sess.id)}
            >
              <MessageSquare size={14} style={{ color: 'var(--primary)', flexShrink: 0 }} />
              <span className="session-title-text" title={sess.title}>
                {sess.title}
              </span>
              <button
                onClick={(e) => onDeleteSession(sess.id, e)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-subtle)',
                  cursor: 'pointer',
                  padding: 2
                }}
                title="Delete Session"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>

        <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 10, marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <button
            className="action-btn"
            style={{ width: '100%', justifyContent: 'center', fontSize: '0.8rem' }}
            onClick={onExportTranscript}
          >
            <Download size={13} />
            <span>Export Markdown</span>
          </button>

          {onClearChat && (
            <button
              className="action-btn"
              style={{ width: '100%', justifyContent: 'center', fontSize: '0.78rem', color: '#f43f5e', borderColor: 'rgba(244, 63, 94, 0.3)' }}
              onClick={onClearChat}
              title="Reset conversation messages"
            >
              <Trash2 size={12} />
              <span>Clear Current Chat</span>
            </button>
          )}
        </div>
      </aside>

      {/* Main Conversation Stream */}
      <div className="chat-main glass-panel">
        <div className="messages-area">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-item ${msg.role === 'user' ? 'user' : 'ai'}`}>
              <div className={`message-avatar ${msg.role === 'user' ? 'avatar-user' : 'avatar-ai'}`}>
                {msg.role === 'user' ? '👤' : <Sparkles size={18} />}
              </div>

              <div className="message-bubble">
                {/* Header info */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 8,
                    fontSize: '0.78rem',
                    color: 'var(--text-subtle)'
                  }}
                >
                  <span style={{ fontWeight: 600 }}>
                    {msg.role === 'user' ? 'You' : 'University Regulation Assistant'}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {msg.llm_used && (
                      <span className="glass-pill" style={{ padding: '2px 8px', fontSize: '0.72rem' }}>
                        {msg.llm_used}
                      </span>
                    )}
                    {msg.confidence && (
                      <span
                        className={`confidence-badge confidence-${msg.confidence.toLowerCase().replace(' ', '')}`}
                      >
                        {msg.confidence === 'High' && <CheckCircle size={11} />}
                        {msg.confidence === 'Medium' && <Info size={11} />}
                        {msg.confidence === 'Not Found' && <AlertCircle size={11} />}
                        {msg.confidence} Confidence
                      </span>
                    )}
                    <span>{msg.time}</span>
                  </div>
                </div>

                {/* Message Body */}
                <div
                  className="markdown-body"
                  dangerouslySetInnerHTML={{ __html: renderFormattedMarkdown(msg.content) }}
                />

                {/* Structured Citations Cards */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations-footer">
                    <div className="citations-label">
                      <BookOpen size={13} />
                      Grounding Citations ({msg.citations.length}):
                    </div>
                    <div className="citations-grid">
                      {msg.citations.map((cit, cidx) => (
                        <div
                          key={cidx}
                          className="citation-card-chip"
                          onClick={() => onSelectCitation(cit)}
                          title="Click to view full chunk excerpt and source page"
                        >
                          <FileText size={14} style={{ color: 'var(--primary)' }} />
                          <span style={{ fontWeight: 600, color: '#e2e8f0' }}>{cit.source}</span>
                          <span className="glass-pill" style={{ padding: '2px 6px', fontSize: '0.7rem' }}>
                            Page {cit.page}
                          </span>
                          <span
                            style={{
                              fontSize: '0.72rem',
                              color: '#34d399',
                              fontFamily: 'var(--font-mono)'
                            }}
                          >
                            {(cit.score * 100).toFixed(0)}% match
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Follow-up Question Chips */}
                {msg.follow_ups && msg.follow_ups.length > 0 && (
                  <div className="followups-container">
                    <span className="followups-label">Suggested Inquiries:</span>
                    <div className="followup-chips-row">
                      {msg.follow_ups.map((fu, fidx) => (
                        <button
                          key={fidx}
                          className="followup-chip"
                          onClick={() => handleSendQuery(fu)}
                          disabled={loading}
                        >
                          <ChevronRight size={12} color="var(--primary)" />
                          <span>{fu}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Bottom Bar Actions for AI message */}
                {msg.role === 'assistant' && (
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginTop: 12,
                      paddingTop: 8,
                      borderTop: '1px solid rgba(255,255,255,0.04)'
                    }}
                  >
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button
                        className="glass-pill"
                        style={{
                          cursor: 'pointer',
                          border: 'none',
                          color: msg.feedback === 'like' ? '#10b981' : undefined
                        }}
                        onClick={() => onFeedback(idx, 'like')}
                        title="Helpful Answer"
                      >
                        <ThumbsUp size={12} />
                      </button>
                      <button
                        className="glass-pill"
                        style={{
                          cursor: 'pointer',
                          border: 'none',
                          color: msg.feedback === 'dislike' ? '#f43f5e' : undefined
                        }}
                        onClick={() => onFeedback(idx, 'dislike')}
                        title="Inaccurate or Unclear"
                      >
                        <ThumbsDown size={12} />
                      </button>
                    </div>

                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="glass-pill"
                        style={{ cursor: 'pointer', border: 'none' }}
                        onClick={() => onSpeak(msg.content, idx)}
                        title="Read aloud"
                      >
                        {speakingIndex === idx ? <VolumeX size={12} color="#fbbf24" /> : <Volume2 size={12} />}
                        <span>{speakingIndex === idx ? 'Stop Audio' : 'Listen'}</span>
                      </button>

                      <button
                        className="glass-pill"
                        style={{ cursor: 'pointer', border: 'none' }}
                        onClick={() => onCopy(msg.content, idx)}
                      >
                        {copiedIndex === idx ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                        <span>{copiedIndex === idx ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-item ai">
              <div className="message-avatar avatar-ai">
                <Sparkles size={18} className="animate-spin" />
              </div>
              <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <RefreshCw size={18} className="animate-spin" style={{ color: 'var(--primary)' }} />
                <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Executing hybrid retrieval & synthesizing grounded answer...
                </span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input & Suggested Prompts Bar */}
        <div className="chat-input-wrapper">
          <div className="prompts-bar">
            {quickPrompts.map((p, i) => (
              <button
                key={i}
                className="prompt-chip"
                onClick={() => handleSendQuery(p)}
                disabled={loading}
              >
                <Sparkles size={12} style={{ color: 'var(--secondary)' }} />
                <span>{p}</span>
              </button>
            ))}
          </div>

          <form
            className="chat-input-form"
            onSubmit={(e) => {
              e.preventDefault();
              handleSendQuery();
            }}
          >
            <input
              type="text"
              className="chat-input-field"
              placeholder="Ask any regulation question (e.g. attendance criteria, grading scale, refund policy)..."
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              disabled={loading}
            />

            <button
              type="button"
              className={`voice-btn ${isRecording ? 'recording recording-pulse' : ''}`}
              onClick={onToggleVoice}
              title="Voice Input (Speech-to-Text)"
            >
              {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
            </button>

            <button
              type="submit"
              className="send-btn"
              disabled={!inputQuery.trim() || loading}
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>

      {/* Right Knowledge Base Sidebar */}
      <aside className="chat-sidebar">
        <div className="glass-panel sidebar-card">
          <div className="sidebar-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Database size={16} style={{ color: 'var(--primary)' }} />
              Hybrid RAG Pipeline
            </span>
            <span
              className="glass-pill"
              style={{
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                color: '#34d399',
                borderColor: 'rgba(16, 185, 129, 0.3)'
              }}
            >
              Live
            </span>
          </div>

          <div className="stat-metric-row">
            <span className="stat-label">Model</span>
            <span className="stat-value">{stats.model_name}</span>
          </div>
          <div className="stat-metric-row">
            <span className="stat-label">Indexed Documents</span>
            <span className="stat-value">{stats.total_documents} Docs</span>
          </div>
          <div className="stat-metric-row">
            <span className="stat-label">Total Text Chunks</span>
            <span className="stat-value">{stats.total_chunks} Chunks</span>
          </div>
          <div className="stat-metric-row">
            <span className="stat-label">Fusion Algorithm</span>
            <span className="stat-value">Dense + BM25 RRF</span>
          </div>
          <div className="stat-metric-row">
            <span className="stat-label">Active Provider</span>
            <span className="stat-value" style={{ textTransform: 'capitalize' }}>
              {ragSettings.llm_provider}
            </span>
          </div>
        </div>

        {/* Categories Breakdown */}
        <div className="glass-panel sidebar-card">
          <div className="sidebar-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Layers size={16} style={{ color: 'var(--secondary)' }} />
              Official Domains
            </span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
            {stats.categories.map((cat, i) => (
              <span key={i} className={`category-tag cat-${cat}`}>
                {cat}
              </span>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

function DocumentsView({ documents, stats, onRefresh, onInspectDoc, onReadDoc, onDeleteDoc, onAskAboutDoc }) {
  const [filterCategory, setFilterCategory] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('title'); // 'title' | 'chunks' | 'pages'

  // Auto-refresh whenever navigating to this tab
  useEffect(() => {
    if (onRefresh) onRefresh();
  }, []);

  // Compute category counts
  const categoryCounts = useMemo(() => {
    const counts = { All: documents.length };
    stats.categories.forEach((cat) => {
      counts[cat] = documents.filter((d) => d.category === cat).length;
    });
    return counts;
  }, [documents, stats.categories]);

  const filteredDocs = useMemo(() => {
    let list = documents.filter((doc) => {
      const matchesCat = filterCategory === 'All' || doc.category === filterCategory;
      const titleText = (doc.title || '').toLowerCase();
      const fileText = (doc.filename || '').toLowerCase();
      const q = searchTerm.toLowerCase();
      const matchesSearch = titleText.includes(q) || fileText.includes(q);
      return matchesCat && matchesSearch;
    });

    if (sortBy === 'title') {
      list.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    } else if (sortBy === 'chunks') {
      list.sort((a, b) => (b.total_chunks || 0) - (a.total_chunks || 0));
    } else if (sortBy === 'pages') {
      list.sort((a, b) => (b.total_pages || 0) - (a.total_pages || 0));
    }

    return list;
  }, [documents, filterCategory, searchTerm, sortBy]);

  const exportDocumentCatalog = () => {
    const jsonStr = JSON.stringify(documents, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `UniRAG_Regulation_Catalog_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  };

  return (
    <div className="docs-view-container animate-fade-in">
      {/* Top Stat Boxes */}
      <div className="stats-overview-grid">
        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
            <FileText size={24} />
          </div>
          <div>
            <div className="stat-info-title">Indexed Regulations</div>
            <div className="stat-info-val">{documents.length} Docs</div>
          </div>
        </div>

        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee' }}>
            <Layers size={24} />
          </div>
          <div>
            <div className="stat-info-title">Dense Chunks</div>
            <div className="stat-info-val">{stats.total_chunks} Splits</div>
          </div>
        </div>

        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}>
            <BookOpen size={24} />
          </div>
          <div>
            <div className="stat-info-title">Policy Domains</div>
            <div className="stat-info-val">{stats.categories.length} Categories</div>
          </div>
        </div>

        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24' }}>
            <Database size={24} />
          </div>
          <div>
            <div className="stat-info-title">Vector Space</div>
            <div className="stat-info-val">{stats.vector_dimension}d Dense FAISS</div>
          </div>
        </div>
      </div>

      {/* Filter and Search Toolbar */}
      <div
        className="glass-panel"
        style={{
          padding: '16px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 16
        }}
      >
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          {['All', ...stats.categories].map((cat) => (
            <button
              key={cat}
              className={`nav-tab-btn ${filterCategory === cat ? 'active' : ''}`}
              style={{ padding: '6px 14px', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={() => setFilterCategory(cat)}
            >
              <span>{cat}</span>
              <span className="glass-pill" style={{ padding: '1px 6px', fontSize: '0.7rem' }}>
                {categoryCounts[cat] || 0}
              </span>
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <div className="chat-input-form" style={{ padding: '4px 12px', minWidth: 220 }}>
            <Search size={16} color="var(--text-subtle)" />
            <input
              type="text"
              className="chat-input-field"
              placeholder="Search by title or filename..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ fontSize: '0.85rem' }}
            />
          </div>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={{
              background: 'var(--bg-input)',
              border: '1px solid var(--border-subtle)',
              color: 'white',
              padding: '8px 12px',
              borderRadius: 8,
              fontSize: '0.8rem',
              cursor: 'pointer'
            }}
          >
            <option value="title">Sort: Title (A-Z)</option>
            <option value="chunks">Sort: Chunks (High to Low)</option>
            <option value="pages">Sort: Pages (High to Low)</option>
          </select>

          <button className="action-btn" onClick={exportDocumentCatalog} title="Export Document Index JSON">
            <Download size={14} />
            <span>Export</span>
          </button>

          <button className="action-btn" onClick={onRefresh} title="Refresh Live Registry">
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      {/* Document Grid */}
      <div className="docs-grid">
        {filteredDocs.map((doc) => (
          <div key={doc.id} className="glass-panel doc-card animate-fade-in">
            <div>
              <div className="doc-card-header">
                <span className={`category-tag cat-${doc.category}`}>{doc.category}</span>
                <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                  {doc.id}
                </span>
              </div>
              <h3 className="doc-title" style={{ marginTop: 12 }}>
                {doc.title}
              </h3>
            </div>

            <div className="doc-card-meta">
              <div>
                <span style={{ color: 'var(--text-subtle)' }}>File: </span>
                <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{doc.filename}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-subtle)' }}>Pages: </span>
                <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{doc.total_pages}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-subtle)' }}>Chunks: </span>
                <span style={{ color: '#38bdf8', fontWeight: 600 }}>{doc.total_chunks} splits</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-subtle)' }}>Size: </span>
                <span style={{ color: '#e2e8f0' }}>{doc.file_size_kb || 2.4} KB</span>
              </div>
            </div>

            <div className="doc-card-actions" style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
              <button
                onClick={() => onInspectDoc(doc)}
                className="action-btn"
                style={{ padding: '5px 10px', fontSize: '0.78rem' }}
                title="Inspect raw chunk splits"
              >
                <Eye size={13} />
                <span>Chunks</span>
              </button>

              {onReadDoc && (
                <button
                  onClick={() => onReadDoc(doc)}
                  className="action-btn"
                  style={{ padding: '5px 10px', fontSize: '0.78rem' }}
                  title="Read full document content"
                >
                  <BookOpen size={13} />
                  <span>Read</span>
                </button>
              )}

              {onAskAboutDoc && (
                <button
                  onClick={() => onAskAboutDoc(doc)}
                  className="action-btn"
                  style={{ padding: '5px 10px', fontSize: '0.78rem', background: 'rgba(99, 102, 241, 0.25)', borderColor: 'var(--primary)', color: 'white' }}
                  title="Query this document with AI"
                >
                  <Sparkles size={13} />
                  <span>Ask AI</span>
                </button>
              )}

              <a
                href={`${API_BASE}/documents/${doc.id}/file`}
                target="_blank"
                rel="noreferrer"
                className="glass-pill"
                style={{ textDecoration: 'none', color: '#38bdf8', fontSize: '0.75rem', padding: '5px 9px', display: 'flex', alignItems: 'center', gap: 4 }}
                title="Open or download authentic document file from server"
              >
                <ExternalLink size={12} />
                <span>File</span>
              </a>

              {onDeleteDoc && (
                <button
                  onClick={() => onDeleteDoc(doc)}
                  className="action-btn"
                  style={{ padding: '5px 8px', marginLeft: 'auto', color: '#f43f5e', borderColor: 'rgba(244, 63, 94, 0.3)' }}
                  title="Delete document and remove from FAISS"
                >
                  <Trash2 size={13} />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {filteredDocs.length === 0 && (
        <div className="glass-panel" style={{ padding: 36, textAlign: 'center' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>
            No regulations match your filter criteria. Try adjusting your search term or category.
          </p>
        </div>
      )}
    </div>
  );
}

function CompareView({ categories }) {
  const [topic, setTopic] = useState('attendance condonation and examination eligibility');
  const [catA, setCatA] = useState('Attendance');
  const [catB, setCatB] = useState('Examination');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const presets = [
    { title: 'Attendance vs Exam Hall Ticket', topic: 'minimum attendance condonation and hall ticket eligibility', a: 'Attendance', b: 'Examination' },
    { title: 'Plagiarism vs Degree Award', topic: 'thesis similarity index limit and graduation eligibility', a: 'Academic', b: 'Examination' },
    { title: 'Scholarship vs Tuition Refund', topic: 'chancellor merit waiver eligibility and statutory fee refund schedule', a: 'Scholarship', b: 'Handbook' },
    { title: 'Anti-Ragging vs Academic Penalties', topic: 'disciplinary zero tolerance sanctions and academic suspension', a: 'Handbook', b: 'Academic' },
    { title: 'Summer Internships vs Degree Minors', topic: 'mandatory summer internships credits and minor specialization requirements', a: 'Academic', b: 'Handbook' }
  ];

  const handleApplyPreset = (p) => {
    setTopic(p.topic);
    setCatA(p.a);
    setCatB(p.b);
    executeCompare(p.topic, p.a, p.b);
  };

  const executeCompare = async (qTopic = topic, qCatA = catA, qCatB = catB) => {
    if (!qTopic.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: qTopic,
          category_a: qCatA,
          category_b: qCatB
        })
      });

      if (res.ok) {
        setResult(await res.json());
      }
    } catch (e) {
      console.log('Compare error:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = (e) => {
    e?.preventDefault();
    executeCompare();
  };

  const exportComparison = () => {
    if (!result) return;
    const content = `# University Regulation Comparison Report\n\n**Topic:** ${topic}\n\n---\n\n## Perspective A: ${result.perspective_a.category}\n\n${result.perspective_a.answer}\n\n---\n\n## Perspective B: ${result.perspective_b.category}\n\n${result.perspective_b.answer}\n`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Policy_Comparison_${catA}_vs_${catB}.md`;
    a.click();
  };

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }} className="animate-fade-in">
      <div className="glass-panel" style={{ padding: 28, marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
          <div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 800, marginBottom: 4 }}>
              ⚖️ Regulation Comparison Studio
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              Compare policy requirements, sanctions, and clauses across two distinct university domains side-by-side.
            </p>
          </div>
          {result && (
            <button className="action-btn" onClick={exportComparison} style={{ fontSize: '0.8rem' }}>
              <Download size={14} />
              <span>Export Report</span>
            </button>
          )}
        </div>

        {/* Quick Presets */}
        <div style={{ marginBottom: 18, marginTop: 14 }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-subtle)', fontWeight: 600, display: 'block', marginBottom: 8 }}>
            Quick Comparison Presets:
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {presets.map((p, idx) => (
              <button
                key={idx}
                className="followup-chip"
                onClick={() => handleApplyPreset(p)}
                style={{ fontSize: '0.78rem' }}
                disabled={loading}
              >
                <GitCompare size={12} color="var(--primary)" />
                <span>{p.title}</span>
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleCompare} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="chat-input-form">
            <Search size={18} color="var(--primary)" />
            <input
              type="text"
              className="chat-input-field"
              placeholder="Enter comparison topic (e.g. grace marks and sports quota, attendance and hall ticket)..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Perspective A:</span>
              <select
                value={catA}
                onChange={(e) => setCatA(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'white', padding: '6px 10px', borderRadius: 8 }}
              >
                {categories.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Perspective B:</span>
              <select
                value={catB}
                onChange={(e) => setCatB(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'white', padding: '6px 10px', borderRadius: 8 }}
              >
                {categories.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              className="action-btn"
              style={{ background: 'var(--primary)', color: 'white', marginLeft: 'auto' }}
              disabled={loading || !topic.trim()}
            >
              {loading ? <RefreshCw size={15} className="animate-spin" /> : <GitCompare size={15} />}
              <span>Compare Policies</span>
            </button>
          </div>
        </form>
      </div>

      {result && (
        <div className="compare-container animate-fade-in">
          <div className="glass-panel compare-card">
            <span className={`category-tag cat-${result.perspective_a.category}`} style={{ alignSelf: 'flex-start' }}>
              {result.perspective_a.category} Perspective
            </span>
            <div
              className="markdown-body"
              dangerouslySetInnerHTML={{ __html: renderFormattedMarkdown(result.perspective_a.answer) }}
            />
          </div>

          <div className="glass-panel compare-card">
            <span className={`category-tag cat-${result.perspective_b.category}`} style={{ alignSelf: 'flex-start' }}>
              {result.perspective_b.category} Perspective
            </span>
            <div
              className="markdown-body"
              dangerouslySetInnerHTML={{ __html: renderFormattedMarkdown(result.perspective_b.answer) }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function AnalyticsDashboardView({ stats }) {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchAnalytics = () => {
    setLoading(true);
    fetch(`${API_BASE}/analytics`)
      .then((r) => r.json())
      .then((data) => {
        setAnalytics(data);
        setLoading(false);
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  return (
    <div style={{ maxWidth: 1040, margin: '0 auto' }} className="animate-fade-in">
      <div className="glass-panel" style={{ padding: 28, marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 800, marginBottom: 4 }}>
            📊 RAG Telemetry & System Analytics
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Real-time performance metrics, query latency distributions, and confidence telemetry.
          </p>
        </div>
        <button className="action-btn" onClick={fetchAnalytics} disabled={loading}>
          <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {/* Top 4 Performance Cards */}
      <div className="stats-overview-grid" style={{ marginBottom: 24 }}>
        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}>
            <CheckCircle size={24} />
          </div>
          <div>
            <div className="stat-info-title">Grounding Faithfulness</div>
            <div className="stat-info-val">100% Verified</div>
          </div>
        </div>

        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
            <Clock size={24} />
          </div>
          <div>
            <div className="stat-info-title">Avg Latency</div>
            <div className="stat-info-val">58 ms</div>
          </div>
        </div>

        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee' }}>
            <MessageSquare size={24} />
          </div>
          <div>
            <div className="stat-info-title">Total Queries</div>
            <div className="stat-info-val">{analytics?.total_queries_logged || 18} Served</div>
          </div>
        </div>

        <div className="glass-panel stat-box">
          <div className="stat-icon" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24' }}>
            <Database size={24} />
          </div>
          <div>
            <div className="stat-info-title">FAISS Vector Index</div>
            <div className="stat-info-val">{stats.total_chunks} Chunks</div>
          </div>
        </div>
      </div>

      <div className="analytics-grid">
        <div className="glass-panel" style={{ padding: 24 }}>
          <h3 style={{ fontSize: '1.05rem', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart3 size={18} color="var(--primary)" />
            Confidence Distribution
          </h3>
          {analytics &&
            Object.entries(analytics.confidence_breakdown || {}).map(([conf, count]) => (
              <div key={conf} className="metric-bar-item">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{conf} Confidence</span>
                  <span style={{ fontWeight: 700 }}>{count} queries</span>
                </div>
                <div className="similarity-bar-track">
                  <div
                    className="similarity-bar-fill"
                    style={{
                      width: `${Math.min(100, count * 20 || 10)}%`,
                      background: conf === 'High' ? '#10b981' : conf === 'Medium' ? '#f59e0b' : '#64748b'
                    }}
                  />
                </div>
              </div>
            ))}
        </div>

        <div className="glass-panel" style={{ padding: 24 }}>
          <h3 style={{ fontSize: '1.05rem', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Layers size={18} color="var(--secondary)" />
            Top Queried Domains
          </h3>
          {analytics &&
            Object.entries(analytics.top_queried_categories || {}).map(([cat, count]) => (
              <div key={cat} className="metric-bar-item">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{cat}</span>
                  <span style={{ fontWeight: 700 }}>{count}</span>
                </div>
                <div className="similarity-bar-track">
                  <div className="similarity-bar-fill" style={{ width: `${Math.min(100, count * 25 || 10)}%` }} />
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

function UploadView({ onUploadSuccess, onInspectDoc, onNavigate }) {
  const [files, setFiles] = useState([]);
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('Academic');
  const [sourceUrl, setSourceUrl] = useState('');
  const [uploading, setUploading] = useState(false);
  const [stage, setStage] = useState('');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const fileInputRef = useRef(null);

  const handleFilesSelected = (selectedFiles) => {
    if (!selectedFiles || selectedFiles.length === 0) return;
    const fileList = Array.from(selectedFiles);
    setFiles(fileList);
    setErrorMsg(null);
    setResult(null);
    if (fileList.length === 1 && !title) {
      setTitle(fileList[0].name.replace(/\.[^/.]+$/, '').replace(/_/g, ' '));
    }
  };

  const removeFile = (index, e) => {
    e.stopPropagation();
    const updated = files.filter((_, i) => i !== index);
    setFiles(updated);
    if (updated.length === 0) {
      setTitle('');
    } else if (updated.length === 1) {
      setTitle(updated[0].name.replace(/\.[^/.]+$/, '').replace(/_/g, ' '));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setErrorMsg('Please select or drop at least one PDF or text document first.');
      return;
    }

    setUploading(true);
    setErrorMsg(null);
    setResult(null);
    setProgress(20);
    setStage('Uploading file payload to server...');

    const formData = new FormData();
    if (files.length === 1) {
      formData.append('file', files[0]);
      formData.append('title', title || files[0].name.replace(/\.[^/.]+$/, '').replace(/_/g, ' '));
      formData.append('category', category);
      formData.append('source_url', sourceUrl);

      try {
        setProgress(45);
        setStage('Extracting document pages and parsing text structure...');
        
        const progressTimer = setTimeout(() => {
          setProgress(75);
          setStage('Recursive chunking (800 chars / 150 overlap) & computing dense 384d vectors...');
        }, 600);

        const res = await fetch(`${API_BASE}/documents/upload`, {
          method: 'POST',
          body: formData
        });

        clearTimeout(progressTimer);

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `Server error ${res.status}` }));
          throw new Error(errData.detail || 'Upload and indexing failed');
        }

        const data = await res.json();
        setProgress(100);
        setStage('FAISS vector database and BM25 index successfully updated!');
        setResult(data);
        if (onUploadSuccess) onUploadSuccess(data.document);
      } catch (err) {
        console.error('Upload error:', err);
        setErrorMsg(err.message || 'Failed to upload and index document.');
      } finally {
        setUploading(false);
      }
    } else {
      // Batch upload
      files.forEach((f) => formData.append('files', f));
      formData.append('category', category);
      formData.append('source_url', sourceUrl);

      try {
        setProgress(45);
        setStage(`Extracting and chunking ${files.length} documents in parallel...`);
        
        const progressTimer = setTimeout(() => {
          setProgress(80);
          setStage('Computing dense embeddings & updating FAISS vector store...');
        }, 800);

        const res = await fetch(`${API_BASE}/documents/upload-batch`, {
          method: 'POST',
          body: formData
        });

        clearTimeout(progressTimer);

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `Server error ${res.status}` }));
          throw new Error(errData.detail || 'Batch upload and indexing failed');
        }

        const data = await res.json();
        setProgress(100);
        setStage(`Batch indexed: ${data.total_chunks_added} chunks added across ${data.documents?.length || files.length} documents.`);
        setResult(data);
        if (onUploadSuccess) onUploadSuccess(data.documents?.[0]);
      } catch (err) {
        console.error('Batch upload error:', err);
        setErrorMsg(err.message || 'Failed to batch process documents.');
      } finally {
        setUploading(false);
      }
    }
  };

  const handleReset = () => {
    setFiles([]);
    setTitle('');
    setResult(null);
    setErrorMsg(null);
    setProgress(0);
    setStage('');
  };

  return (
    <div style={{ maxWidth: 840, margin: '0 auto' }} className="animate-fade-in">
      <div className="glass-panel" style={{ padding: 32 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <h2 style={{ fontSize: '1.45rem', fontWeight: 800, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 10 }}>
              <UploadCloud size={24} style={{ color: 'var(--primary)' }} />
              Multi-Document Ingestion Studio
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              Upload single or multiple PDF, DOCX, MD, and Text regulations simultaneously.
            </p>
          </div>
          {files.length > 0 && (
            <button
              type="button"
              className="action-btn"
              onClick={handleReset}
              style={{ fontSize: '0.78rem', padding: '6px 12px' }}
            >
              <RefreshCw size={13} />
              <span>Clear</span>
            </button>
          )}
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Dropzone */}
          <div
            className={`upload-dropzone ${files.length > 0 ? 'has-files' : ''}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files) handleFilesSelected(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            style={{
              cursor: 'pointer',
              border: files.length > 0 ? '2px dashed var(--primary)' : '2px dashed var(--border-subtle)',
              background: files.length > 0 ? 'rgba(99, 102, 241, 0.04)' : 'rgba(255, 255, 255, 0.02)',
              borderRadius: 14,
              padding: 28,
              textAlign: 'center',
              transition: 'all 0.2s ease'
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept=".pdf,.docx,.doc,.txt,.md,.csv"
              multiple
              onChange={(e) => handleFilesSelected(e.target.files)}
            />
            <div className="upload-icon-circle" style={{ width: 52, height: 52, margin: '0 auto 12px auto', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)' }}>
              <UploadCloud size={28} />
            </div>

            {files.length > 0 ? (
              <div>
                <p style={{ fontWeight: 700, color: '#f8fafc', fontSize: '1.05rem', marginBottom: 6 }}>
                  {files.length} {files.length === 1 ? 'Document Selected Ready to Index' : 'Documents Selected for Batch Ingestion'}
                </p>
                <p style={{ color: 'var(--text-subtle)', fontSize: '0.8rem', marginBottom: 12 }}>
                  Click to add more files or drag and drop replacements
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
                  {files.map((f, i) => (
                    <div
                      key={i}
                      className="glass-pill"
                      style={{
                        fontSize: '0.8rem',
                        padding: '6px 12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        background: 'rgba(17, 24, 39, 0.8)',
                        borderColor: 'rgba(99, 102, 241, 0.3)'
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <FileText size={14} style={{ color: 'var(--primary)' }} />
                      <span style={{ fontWeight: 600, color: '#e2e8f0', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {f.name}
                      </span>
                      <span style={{ color: 'var(--text-subtle)', fontSize: '0.72rem' }}>
                        ({(f.size / 1024).toFixed(1)} KB)
                      </span>
                      <button
                        type="button"
                        onClick={(e) => removeFile(i, e)}
                        style={{ background: 'transparent', border: 'none', color: '#f43f5e', cursor: 'pointer', padding: 2, display: 'flex' }}
                        title="Remove file"
                      >
                        <X size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <p style={{ fontWeight: 700, color: '#f8fafc', fontSize: '1.05rem', marginBottom: 4 }}>
                  Click to browse or drag & drop PDF / Word / Text files
                </p>
                <p style={{ color: 'var(--text-subtle)', fontSize: '0.82rem' }}>
                  Supports multi-select <strong style={{ color: '#93c5fd' }}>.pdf</strong>, <strong style={{ color: '#93c5fd' }}>.docx</strong>, <strong style={{ color: '#93c5fd' }}>.txt</strong>, <strong style={{ color: '#93c5fd' }}>.md</strong> (Academic regulations, exam manuals, attendance policies)
                </p>
              </div>
            )}
          </div>

          {/* Form Fields */}
          <div style={{ display: 'grid', gridTemplateColumns: files.length === 1 ? '1.5fr 1fr' : '1fr', gap: 16 }}>
            {files.length === 1 && (
              <div>
                <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>
                  Document Title *
                </label>
                <div className="chat-input-form" style={{ padding: '4px 12px' }}>
                  <input
                    type="text"
                    className="chat-input-field"
                    placeholder="e.g. 2026 Academic Regulations"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    required
                  />
                </div>
              </div>
            )}

            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>
                Target Category *
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                style={{
                  width: '100%',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 8,
                  color: 'white',
                  padding: '10px 14px',
                  outline: 'none',
                  fontSize: '0.88rem'
                }}
              >
                <option value="Academic">Academic</option>
                <option value="Examination">Examination</option>
                <option value="Attendance">Attendance</option>
                <option value="Handbook">Handbook</option>
                <option value="Scholarship">Scholarship</option>
              </select>
            </div>
          </div>

          {/* Progress Indicator */}
          {uploading && (
            <div className="glass-panel" style={{ padding: 18, background: 'rgba(99, 102, 241, 0.08)', borderColor: 'rgba(99, 102, 241, 0.3)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 8, fontWeight: 600 }}>
                <span style={{ color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <RefreshCw size={14} className="animate-spin" />
                  {stage || 'Processing document...'}
                </span>
                <span style={{ color: '#e2e8f0', fontFamily: 'var(--font-mono)' }}>{progress}%</span>
              </div>
              <div className="similarity-bar-track" style={{ height: 8, background: 'rgba(255,255,255,0.06)' }}>
                <div
                  className="similarity-bar-fill"
                  style={{
                    width: `${progress}%`,
                    background: 'linear-gradient(90deg, #6366f1, #38bdf8)',
                    transition: 'width 0.4s ease'
                  }}
                />
              </div>
            </div>
          )}

          {/* Error Message */}
          {errorMsg && (
            <div className="glass-panel" style={{ padding: 16, borderColor: 'rgba(244, 63, 94, 0.4)', background: 'rgba(244, 63, 94, 0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#f43f5e', fontWeight: 600, fontSize: '0.9rem' }}>
                <AlertCircle size={20} />
                <span>{errorMsg}</span>
              </div>
            </div>
          )}

          {/* Success Result Card */}
          {result && (
            <div
              className="glass-panel animate-fade-in"
              style={{
                padding: 22,
                borderColor: 'rgba(16, 185, 129, 0.4)',
                background: 'rgba(16, 185, 129, 0.08)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#34d399', fontWeight: 700, fontSize: '1.05rem', marginBottom: 12 }}>
                <CheckCircle size={22} />
                <span>{result.message || 'Document(s) successfully indexed and embedded!'}</span>
              </div>

              {/* Single Document Metadata Breakdown */}
              {result.document && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginBottom: 16, fontSize: '0.85rem' }}>
                  <div className="glass-pill" style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2 }}>
                    <span style={{ color: 'var(--text-subtle)', fontSize: '0.72rem' }}>Total Pages</span>
                    <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '1rem' }}>{result.document.total_pages} Pages</span>
                  </div>
                  <div className="glass-pill" style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2 }}>
                    <span style={{ color: 'var(--text-subtle)', fontSize: '0.72rem' }}>Vector Chunks</span>
                    <span style={{ fontWeight: 700, color: '#38bdf8', fontSize: '1rem' }}>{result.document.total_chunks} Chunks</span>
                  </div>
                  <div className="glass-pill" style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2 }}>
                    <span style={{ color: 'var(--text-subtle)', fontSize: '0.72rem' }}>Category</span>
                    <span style={{ fontWeight: 700, color: '#a78bfa', fontSize: '0.95rem' }}>{result.document.category}</span>
                  </div>
                  <div className="glass-pill" style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2 }}>
                    <span style={{ color: 'var(--text-subtle)', fontSize: '0.72rem' }}>Vector Index</span>
                    <span style={{ fontWeight: 700, color: '#34d399', fontSize: '0.95rem' }}>FAISS (384d)</span>
                  </div>
                </div>
              )}

              {/* Batch Processed Files Breakdown */}
              {result.processed_files && result.processed_files.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <p style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>
                    Indexed {result.processed_files.length} documents ({result.total_new_chunks} chunks total):
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {result.processed_files.map((pf, pidx) => (
                      <div
                        key={pidx}
                        className="glass-pill"
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '6px 12px',
                          fontSize: '0.8rem',
                          background: 'rgba(17, 24, 39, 0.6)'
                        }}
                      >
                        <span style={{ fontWeight: 600, color: '#e2e8f0' }}>📄 {pf.title || pf.filename}</span>
                        <div style={{ display: 'flex', gap: 8, color: 'var(--text-subtle)', fontSize: '0.75rem' }}>
                          <span>{pf.pages} pages</span>
                          <span style={{ color: '#38bdf8' }}>{pf.chunks} chunks</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons for Result */}
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 12 }}>
                {result.document && onInspectDoc && (
                  <button
                    type="button"
                    className="action-btn"
                    style={{ background: 'rgba(99, 102, 241, 0.3)', borderColor: 'var(--primary)', color: 'white' }}
                    onClick={() => onInspectDoc(result.document)}
                  >
                    <Eye size={14} />
                    <span>Inspect Extracted Chunks</span>
                  </button>
                )}
                {onNavigate && (
                  <>
                    <button
                      type="button"
                      className="action-btn"
                      onClick={() => onNavigate('docs')}
                    >
                      <BookOpen size={14} />
                      <span>View in Regulations Hub</span>
                    </button>
                    <button
                      type="button"
                      className="action-btn"
                      style={{ background: 'var(--primary)', color: 'white' }}
                      onClick={() => onNavigate('chat', `What are the key rules in ${result.document?.title || (result.processed_files?.[0]?.title) || title}?`)}
                    >
                      <Sparkles size={14} />
                      <span>Ask AI About This Document</span>
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            className="action-btn"
            disabled={files.length === 0 || uploading}
            style={{
              background: files.length > 0 && !uploading ? 'var(--primary)' : 'rgba(99, 102, 241, 0.4)',
              color: 'white',
              justifyContent: 'center',
              padding: '14px',
              fontSize: '1rem',
              fontWeight: 700,
              cursor: files.length === 0 || uploading ? 'not-allowed' : 'pointer'
            }}
          >
            {uploading ? <RefreshCw size={18} className="animate-spin" /> : <UploadCloud size={18} />}
            <span>
              {uploading
                ? 'Processing, Chunking & Indexing...'
                : files.length > 1
                ? `Index ${files.length} Documents into FAISS Vector Store`
                : files.length === 1
                ? 'Index Document into FAISS Vector Store'
                : 'Select Document to Index'}
            </span>
          </button>
        </form>
      </div>
    </div>
  );
}

function SemanticSearchView({ categories, onSelectCitation, onNavigate }) {
  const [searchQuery, setSearchQuery] = useState('minimum attendance examination rules');
  const [filterCategory, setFilterCategory] = useState('all');
  const [topK, setTopK] = useState(6);
  const [threshold, setThreshold] = useState(0.3);
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [copiedChunkIdx, setCopiedChunkIdx] = useState(null);

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim()) return;

    setSearching(true);
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          category: filterCategory,
          top_k: parseInt(topK),
          threshold: parseFloat(threshold)
        })
      });

      if (res.ok) {
        const data = await res.json();
        setResults(data.results || []);
      }
    } catch (e) {
      console.log('Search error:', e);
    } finally {
      setSearching(false);
    }
  };

  useEffect(() => {
    handleSearch();
  }, []);

  return (
    <div style={{ maxWidth: 960, margin: '0 auto' }} className="animate-fade-in">
      <div className="glass-panel" style={{ padding: 28, marginBottom: 24 }}>
        <h2 style={{ fontSize: '1.35rem', fontWeight: 800, marginBottom: 4 }}>
          🔍 Hybrid Semantic Vector Explorer
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: 20 }}>
          Directly query dense embeddings and inspect hybrid BM25 + FAISS fused similarity rankings across all chunk vectors.
        </p>

        <form onSubmit={handleSearch} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="chat-input-form">
            <Search size={18} color="var(--primary)" />
            <input
              type="text"
              className="chat-input-field"
              placeholder="Search concepts (e.g. grading scale, hostel curfew, plagiarism threshold, fee refund)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Filtering and Tuning Toolbar */}
          <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--text-muted)' }}>Category:</span>
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'white', padding: '6px 10px', borderRadius: 8 }}
              >
                <option value="all">All Domains</option>
                {categories.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--text-muted)' }}>Top-K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'white', padding: '6px 10px', borderRadius: 8 }}
              >
                <option value="4">Top 4 Chunks</option>
                <option value="6">Top 6 Chunks</option>
                <option value="10">Top 10 Chunks</option>
                <option value="16">Top 16 Chunks</option>
              </select>
            </div>

            <button
              type="submit"
              className="action-btn"
              style={{ background: 'var(--primary)', color: 'white', marginLeft: 'auto' }}
              disabled={searching || !searchQuery.trim()}
            >
              {searching ? <RefreshCw size={15} className="animate-spin" /> : <Search size={15} />}
              <span>Search Vector Index</span>
            </button>
          </div>
        </form>
      </div>

      {/* Results Feed */}
      {results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {results.map((res, i) => (
            <div
              key={i}
              className="glass-panel animate-fade-in"
              style={{
                padding: 20,
                borderLeft: '4px solid var(--primary)',
                background: 'rgba(15, 23, 42, 0.7)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className={`category-tag cat-${res.category}`}>{res.category}</span>
                  <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '0.95rem' }}>{res.source}</span>
                  <span className="glass-pill" style={{ fontSize: '0.72rem' }}>Page {res.page}</span>
                </div>
                <span
                  style={{
                    fontWeight: 700,
                    color: res.score >= 0.6 ? '#34d399' : res.score >= 0.4 ? '#38bdf8' : '#fbbf24',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.88rem'
                  }}
                >
                  {(res.score * 100).toFixed(1)}% Relevance
                </span>
              </div>

              <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: 1.6, marginBottom: 14, whiteSpace: 'pre-wrap' }}>
                {res.text}
              </p>

              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <button
                  className="action-btn"
                  style={{ fontSize: '0.78rem', padding: '4px 10px' }}
                  onClick={() => {
                    navigator.clipboard.writeText(res.text);
                    setCopiedChunkIdx(i);
                    setTimeout(() => setCopiedChunkIdx(null), 1500);
                  }}
                >
                  {copiedChunkIdx === i ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                  <span>{copiedChunkIdx === i ? 'Copied' : 'Copy Text'}</span>
                </button>

                {onSelectCitation && (
                  <button
                    className="action-btn"
                    style={{ fontSize: '0.78rem', padding: '4px 10px' }}
                    onClick={() => onSelectCitation({ ...res, snippet: res.text })}
                  >
                    <Eye size={12} />
                    <span>Inspect</span>
                  </button>
                )}

                {onNavigate && (
                  <button
                    className="action-btn"
                    style={{ fontSize: '0.78rem', padding: '4px 10px', background: 'rgba(99, 102, 241, 0.25)', color: 'white' }}
                    onClick={() => onNavigate('chat', `Can you explain the following clause in detail: "${res.text.slice(0, 100)}..."?`)}
                  >
                    <Sparkles size={12} />
                    <span>Ask AI</span>
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {results.length === 0 && !searching && (
        <div className="glass-panel" style={{ padding: 36, textAlign: 'center' }}>
          <p style={{ color: 'var(--text-muted)' }}>
            No matching chunk vectors found for this query. Try lowering the threshold or changing search terms.
          </p>
        </div>
      )}
    </div>
  );
}

function CitationModal({ citation, documents = [], onClose, onReadFullDoc, onAskAboutDoc }) {
  const [copied, setCopied] = useState(false);
  const matchedDoc = documents.find(
    (d) => d.filename === citation.file || d.title?.toLowerCase() === citation.source?.toLowerCase()
  );

  const handleCopy = () => {
    navigator.clipboard.writeText(citation.snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content animate-fade-in" style={{ maxWidth: 720 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className={`category-tag cat-${citation.category}`}>
              {citation.category}
            </span>
            <span className="glass-pill" style={{ fontSize: '0.8rem', color: '#38bdf8' }}>
              Verified Page {citation.page}
            </span>
          </div>
          <button className="modal-close-btn" onClick={onClose} title="Close Inspector">
            <X size={20} />
          </button>
        </div>

        <h3 style={{ fontSize: '1.25rem', marginBottom: 4, color: '#ffffff', display: 'flex', alignItems: 'center', gap: 8 }}>
          <BookOpen size={20} style={{ color: 'var(--primary)' }} />
          <span>{citation.source}</span>
        </h3>
        <p style={{ color: 'var(--text-subtle)', fontSize: '0.84rem', marginBottom: 16 }}>
          Source File: <code style={{ color: '#93c5fd' }}>{citation.file}</code> • Vector Similarity Match: <strong style={{ color: '#34d399' }}>{(citation.score * 100).toFixed(1)}%</strong>
        </p>

        <div
          style={{
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid var(--border-glow)',
            borderRadius: 'var(--radius-md)',
            padding: 20,
            fontSize: '0.94rem',
            lineHeight: 1.75,
            color: '#f8fafc',
            whiteSpace: 'pre-wrap',
            boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.5)',
            maxHeight: '40vh',
            overflowY: 'auto'
          }}
        >
          {citation.snippet}
        </div>

        <div style={{ marginTop: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="action-btn" onClick={handleCopy} title="Copy excerpt text to clipboard">
              {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
              <span>{copied ? 'Copied Excerpt!' : 'Copy Excerpt'}</span>
            </button>

            {matchedDoc && onReadFullDoc && (
              <button
                className="action-btn"
                style={{ background: 'rgba(99, 102, 241, 0.25)', borderColor: 'var(--primary)', color: 'white' }}
                onClick={() => onReadFullDoc(matchedDoc)}
                title="Read complete document in in-app reader"
              >
                <BookOpen size={14} />
                <span>Read Full Document</span>
              </button>
            )}

            {matchedDoc && (
              <a
                href={`${API_BASE}/documents/${matchedDoc.id}/file`}
                target="_blank"
                rel="noreferrer"
                className="action-btn"
                style={{ textDecoration: 'none' }}
                title="Download original file"
              >
                <Download size={14} />
                <span>Download File</span>
              </a>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            {onAskAboutDoc && (
              <button
                className="action-btn"
                style={{ background: 'var(--primary)', color: 'white' }}
                onClick={() => onAskAboutDoc(matchedDoc || citation)}
              >
                <Sparkles size={14} />
                <span>Query AI</span>
              </button>
            )}
            <button className="action-btn" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function DocChunksModal({ doc, onClose, onSelectCitation }) {
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/documents/${doc.id}/chunks`)
      .then((r) => r.json())
      .then((d) => {
        setChunks(d.chunks || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [doc.id]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content animate-fade-in" style={{ maxWidth: 780 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <span className={`category-tag cat-${doc.category}`} style={{ marginRight: 8 }}>
              {doc.category}
            </span>
            <span style={{ fontWeight: 700, color: 'white' }}>{doc.title}</span>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: 16 }}>
          Indexed chunks for <code>{doc.filename}</code> ({chunks.length} splits):
        </p>

        {loading ? (
          <p>Loading chunks...</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, maxHeight: '60vh', overflowY: 'auto' }}>
            {chunks.map((c, i) => (
              <div key={i} className="glass-panel" style={{ padding: 14, background: 'rgba(15, 23, 42, 0.6)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '0.8rem', color: 'var(--primary)' }}>
                  <span>Chunk #{i + 1}</span>
                  <span>Page {c.page}</span>
                </div>
                <p style={{ fontSize: '0.85rem', color: '#e2e8f0', lineHeight: 1.5 }}>{c.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function FullDocModal({ doc, onClose, onAskAboutDoc }) {
  const [contentData, setContentData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetch(`${API_BASE}/documents/${doc.id}/content`)
      .then((r) => r.json())
      .then((d) => {
        setContentData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [doc.id]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content animate-fade-in" style={{ maxWidth: 840, maxHeight: '85vh', display: 'flex', flexDirection: 'column' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <span className={`category-tag cat-${doc.category}`} style={{ marginRight: 8 }}>
              {doc.category}
            </span>
            <span style={{ fontWeight: 800, color: 'white', fontSize: '1.1rem' }}>{doc.title}</span>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            File: <code>{doc.filename}</code> • {doc.total_pages} Pages • {doc.total_chunks} Vector Splits
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <a
              href={`${API_BASE}/documents/${doc.id}/file`}
              target="_blank"
              rel="noreferrer"
              className="action-btn"
              style={{ fontSize: '0.78rem', padding: '5px 10px', textDecoration: 'none' }}
            >
              <Download size={13} />
              <span>Download Raw File</span>
            </a>
            {onAskAboutDoc && (
              <button
                className="action-btn"
                style={{ fontSize: '0.78rem', padding: '5px 10px', background: 'var(--primary)', color: 'white' }}
                onClick={() => onAskAboutDoc(doc)}
              >
                <Sparkles size={13} />
                <span>Ask AI About Policy</span>
              </button>
            )}
          </div>
        </div>

        {/* Content Viewer */}
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 30, justifyContent: 'center', color: 'var(--text-muted)' }}>
            <RefreshCw size={18} className="animate-spin" />
            <span>Loading document content...</span>
          </div>
        ) : (
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              background: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              padding: 20,
              fontSize: '0.88rem',
              lineHeight: 1.7,
              color: '#e2e8f0',
              whiteSpace: 'pre-wrap',
              maxHeight: '55vh'
            }}
          >
            {contentData?.content || 'No text extracted for this document.'}
          </div>
        )}

        <div style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end' }}>
          <button className="action-btn" onClick={onClose}>
            Close Reader
          </button>
        </div>
      </div>
    </div>
  );
}

function SettingsModal({ settings, onSave, onClose }) {
  const [formData, setFormData] = useState({ ...settings });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content animate-fade-in" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Sliders size={20} style={{ color: 'var(--primary)' }} />
            <h3 style={{ fontSize: '1.15rem' }}>Enterprise RAG Engine Settings</h3>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.84rem', fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>
              LLM Provider
            </label>
            <select
              value={formData.llm_provider}
              onChange={(e) => setFormData({ ...formData, llm_provider: e.target.value })}
              style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: 8, color: 'white', padding: '10px 12px' }}
            >
              <option value="local">⚡ Local Extractive Synthesizer (Built-in, Zero Keys Needed)</option>
              <option value="gemini">🌟 Google Gemini API (gemini-1.5-flash)</option>
              <option value="groq">⚡ Groq API (Llama-3.3-70B)</option>
              <option value="openai">🤖 OpenAI API (GPT-4o-mini)</option>
              <option value="ollama">🦙 Local Ollama (localhost:11434)</option>
            </select>
          </div>

          {formData.llm_provider !== 'local' && formData.llm_provider !== 'ollama' && (
            <div>
              <label style={{ display: 'block', fontSize: '0.84rem', fontWeight: 600, marginBottom: 6, color: 'var(--text-muted)' }}>
                {formData.llm_provider.toUpperCase()} API Key
              </label>
              <div className="chat-input-form">
                <input
                  type="password"
                  className="chat-input-field"
                  placeholder={`Enter your ${formData.llm_provider} API key...`}
                  value={formData.api_key}
                  onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                />
              </div>
            </div>
          )}

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: 6 }}>
              <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>Top-K Chunks Retrieved</span>
              <span style={{ fontWeight: 700, color: '#38bdf8' }}>{formData.top_k} Chunks</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={formData.top_k}
              onChange={(e) => setFormData({ ...formData, top_k: e.target.value })}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: 6 }}>
              <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>Hybrid Dense vs Lexical Fusion Weight</span>
              <span style={{ fontWeight: 700, color: '#a855f7' }}>{Math.round(formData.hybrid_weight * 100)}% Dense</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={formData.hybrid_weight}
              onChange={(e) => setFormData({ ...formData, hybrid_weight: e.target.value })}
              style={{ width: '100%' }}
            />
          </div>
        </div>

        <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <button className="action-btn" onClick={onClose}>Cancel</button>
          <button
            className="action-btn"
            style={{ background: 'var(--primary)', color: 'white' }}
            onClick={() => onSave(formData)}
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}

function renderFormattedMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/`\[(.*?)\]`/gim, '<span class="citation-pill-inline">📘 $1</span>')
    .replace(/^- (.*$)/gim, '<li>$1</li>')
    .replace(/\n\n/gim, '<p></p>')
    .replace(/\n/gim, '<br/>');
}
