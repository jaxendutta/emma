import { useState, useEffect, useRef } from 'react'
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  TypingIndicator,
} from '@chatscope/chat-ui-kit-react'
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css'
import './Chat.css'

const SESSION_ID = 'emma-' + Math.random().toString(36).slice(2, 10)

const OPENERS = [
  'Ready to prep?',
  'Quiz time — what are the classic signs of meningitis?',
  "Did you know sepsis kills more people annually than breast, bowel, and prostate cancer combined?",
  'Can you name the FAST signs of a stroke?',
  "What's the first test you'd order for a suspected pulmonary embolism?",
  'Time is brain — every minute of untreated stroke destroys ~1.9 million neurons.',
  'Did you know DKA can present with a fruity breath smell?',
  'Ask me anything — symptoms, diagnosis, treatment, or how to tell two conditions apart.',
]

export default function Chat() {
  const [open, setOpen]         = useState(false)
  const [messages, setMessages] = useState([])
  const [isTyping, setIsTyping] = useState(false)
  const panelRef  = useRef(null)
  const toggleRef = useRef(null)

  // Welcome message on first open
  useEffect(() => {
    if (open && messages.length === 0) {
      setMessages([{
        message:   OPENERS[Math.floor(Math.random() * OPENERS.length)],
        sender:    'EMMA',
        direction: 'incoming',
        position:  'single',
      }])
    }
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  // Click-outside to close
  useEffect(() => {
    if (!open) return
    const handler = (e) => {
      if (
        panelRef.current  && !panelRef.current.contains(e.target) &&
        toggleRef.current && !toggleRef.current.contains(e.target)
      ) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const handleSend = async (text) => {
    const userMsg = {
      message:   text,
      sender:    'You',
      direction: 'outgoing',
      position:  'single',
    }
    setMessages(prev => [...prev, userMsg])
    setIsTyping(true)

    try {
      const res = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, session_id: SESSION_ID }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        message:   data.answer || 'No answer returned.',
        sender:    'EMMA',
        direction: 'incoming',
        position:  'single',
      }])
    } catch {
      setMessages(prev => [...prev, {
        message:   'Could not reach the EMMA server. Make sure the API is running.',
        sender:    'EMMA',
        direction: 'incoming',
        position:  'single',
      }])
    } finally {
      setIsTyping(false)
    }
  }

  return (
    <>
      <button
        ref={toggleRef}
        className={`emma-toggle${open ? ' open' : ''}`}
        onClick={() => setOpen(o => !o)}
        aria-label="Chat with EMMA"
      >
        {open ? <CloseIcon /> : <ChatIcon />}
      </button>

      <div ref={panelRef} className={`emma-panel${open ? ' open' : ''}`}>
        <div
          className="emma-chat-scope"
          style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
        >
          <div className="emma-chat-header">
            <div className="emma-chat-header-avatar">E</div>
            <div>
              <div className="emma-chat-header-name">EMMA</div>
              <div className="emma-chat-header-sub">Emergency Medicine Mentor</div>
            </div>
          </div>

          <div style={{ flex: 1, minHeight: 0 }}>
            <MainContainer style={{ height: '100%' }}>
              <ChatContainer style={{ height: '100%' }}>
                <MessageList
                  typingIndicator={
                    isTyping
                      ? <TypingIndicator content="EMMA is thinking…" />
                      : null
                  }
                >
                  {messages.map((msg, i) => (
                    <Message key={i} model={msg} />
                  ))}
                </MessageList>
                <MessageInput
                  placeholder="Ask about symptoms, diagnosis, treatment…"
                  onSend={handleSend}
                  attachButton={false}
                  autoFocus={open}
                />
              </ChatContainer>
            </MainContainer>
          </div>
        </div>
      </div>
    </>
  )
}

function ChatIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <path
        d="M3 14.5C3 15.88 4.12 17 5.5 17H16.5C17.88 17 19 15.88 19 14.5V8.5C19 7.12 17.88 6 16.5 6H5.5C4.12 6 3 7.12 3 8.5V14.5Z"
        stroke="white" strokeWidth="1.6"
      />
      <circle cx="8"  cy="11.5" r="1" fill="white" />
      <circle cx="11" cy="11.5" r="1" fill="white" />
      <circle cx="14" cy="11.5" r="1" fill="white" />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path
        d="M4 4L14 14M14 4L4 14"
        stroke="white" strokeWidth="2" strokeLinecap="round"
      />
    </svg>
  )
}
