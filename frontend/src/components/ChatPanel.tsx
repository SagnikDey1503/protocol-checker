import React, { useState, useRef, useEffect } from 'react';
import { Send, ShieldCheck } from 'lucide-react';
import { ChatWebSocket } from '../services/websocket';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  agentType?: string;
  confidence?: number;
  sources?: any[];
}

interface ChatPanelProps {
  sessionId: string;
  experimentId?: string;
  initialMessages?: Message[];
}

export const ChatPanel: React.FC<ChatPanelProps> = ({ sessionId, experimentId, initialMessages = [] }) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [wsClient, setWsClient] = useState<ChatWebSocket | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingResponse]);

  // Clean up WebSocket on change of session
  useEffect(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
    setStreamingResponse('');
    setActiveAgent(null);

    const handlers = {
      onOpen: () => {
        setIsConnected(true);
      },
      onToken: (token: string) => {
        setStreamingResponse((prev) => prev + token);
      },
      onDone: (data: { conversation_id: string; agent_type: string }) => {
        setStreamingResponse((finalContent) => {
          if (finalContent) {
            setMessages((prev) => [
              ...prev,
              {
                id: Math.random().toString(),
                role: 'assistant',
                content: finalContent,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                agentType: data.agent_type,
              },
            ]);
          }
          return '';
        });
        setActiveAgent(null);
      },
      onError: (err: string) => {
        setMessages((prev) => [
          ...prev,
          {
            id: Math.random().toString(),
            role: 'assistant',
            content: `⚠️ Error: ${err}`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);
        setStreamingResponse('');
        setActiveAgent(null);
      },
      onClose: () => {
        setIsConnected(false);
      },
    };

    const client = new ChatWebSocket(sessionId, handlers);
    client.connect();
    setWsClient(client);

    return () => {
      client.disconnect();
    };
  }, [sessionId]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !wsClient) return;

    const userMsg: Message = {
      id: Math.random().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setStreamingResponse('');
    setActiveAgent('Routing Agent...');

    try {
      wsClient.sendMessage(input, experimentId);
      setInput('');
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: Math.random().toString(),
          role: 'assistant',
          content: `⚠️ Connection error: ${err.message}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
      setActiveAgent(null);
    }
  };

  // Simple custom Markdown rendering function
  const renderMarkdown = (text: string) => {
    if (!text) return null;
    const lines = text.split('\n');

    return lines.map((line, idx) => {
      // 1. Headers: #, ##, ###
      if (line.startsWith('### ')) {
        return <h4 key={idx} style={{ marginTop: '14px', marginBottom: '6px', fontWeight: 600, fontSize: '14px', color: 'var(--text-primary)', fontFamily: 'var(--font-title)' }}>{line.slice(4)}</h4>;
      }
      if (line.startsWith('## ')) {
        return <h3 key={idx} style={{ marginTop: '16px', marginBottom: '8px', fontWeight: 600, fontSize: '15px', color: 'var(--text-primary)', fontFamily: 'var(--font-title)' }}>{line.slice(3)}</h3>;
      }
      if (line.startsWith('# ')) {
        return <h2 key={idx} style={{ marginTop: '18px', marginBottom: '10px', fontWeight: 700, fontSize: '17px', color: 'var(--text-primary)', fontFamily: 'var(--font-title)' }}>{line.slice(2)}</h2>;
      }

      // 2. Bullet list items
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        return (
          <li key={idx} style={{ marginLeft: '20px', marginBottom: '4px', listStyleType: 'disc' }}>
            {parseInlineMarkdown(line.trim().slice(2))}
          </li>
        );
      }

      // 3. Numbered lists
      const numberedMatch = line.trim().match(/^(\d+)\.\s+(.*)/);
      if (numberedMatch) {
        return (
          <li key={idx} style={{ marginLeft: '20px', marginBottom: '6px', listStyleType: 'decimal' }}>
            {parseInlineMarkdown(numberedMatch[2])}
          </li>
        );
      }

      // 4. Github-style alerts / blockquotes
      const isNoteAlert = line.trim().startsWith('> [!NOTE]');
      const isWarningAlert = line.trim().startsWith('> [!WARNING]') || line.trim().startsWith('> [!CAUTION]');
      const isImportantAlert = line.trim().startsWith('> [!IMPORTANT]');
      const isQuote = line.trim().startsWith('>');

      if (isNoteAlert || isWarningAlert || isImportantAlert || isQuote) {
        let alertColor = 'var(--text-secondary)';
        let alertBg = 'rgba(255, 255, 255, 0.02)';
        let borderLeftColor = 'var(--text-muted)';
        let displayLine = line;

        if (isNoteAlert) {
          alertBg = 'rgba(99, 102, 241, 0.05)';
          borderLeftColor = 'var(--primary)';
          displayLine = displayLine.replace('> [!NOTE]', '').trim();
        } else if (isWarningAlert) {
          alertBg = 'rgba(245, 158, 11, 0.05)';
          borderLeftColor = 'var(--warning)';
          displayLine = displayLine.replace(/> \[\!(WARNING|CAUTION)\]/, '').trim();
        } else if (isImportantAlert) {
          alertBg = 'rgba(239, 68, 68, 0.05)';
          borderLeftColor = 'var(--danger)';
          displayLine = displayLine.replace('> [!IMPORTANT]', '').trim();
        } else {
          displayLine = displayLine.slice(1).trim();
        }

        return (
          <blockquote key={idx} style={{
            background: alertBg,
            borderLeft: `4px solid ${borderLeftColor}`,
            padding: '8px 14px',
            borderRadius: '0 var(--radius-md) var(--radius-md) 0',
            margin: '10px 0',
            color: alertColor,
            fontSize: '13px',
            lineHeight: 1.4
          }}>
            {parseInlineMarkdown(displayLine)}
          </blockquote>
        );
      }

      // 5. Standard line
      return (
        <p key={idx} style={{ marginBottom: '10px', minHeight: '18px' }}>
          {parseInlineMarkdown(line)}
        </p>
      );
    });
  };

  // Helper to parse inline bold and inline code
  const parseInlineMarkdown = (text: string) => {
    if (!text) return '';
    const parts = [];
    let currentIdx = 0;
    
    // Regex for inline code (`) and bold (**)
    const inlineRegex = /(\*\*.*?\*\*|`.*?`|\[.*?\]\(.*?\))/g;
    let match;

    while ((match = inlineRegex.exec(text)) !== null) {
      const matchStart = match.index;
      const matchText = match[0];
      
      // Push preceding text
      if (matchStart > currentIdx) {
        parts.push(text.substring(currentIdx, matchStart));
      }

      if (matchText.startsWith('**') && matchText.endsWith('**')) {
        parts.push(<strong key={matchStart} style={{ color: '#fff', fontWeight: 600 }}>{matchText.slice(2, -2)}</strong>);
      } else if (matchText.startsWith('`') && matchText.endsWith('`')) {
        parts.push(
          <code key={matchStart} style={{
            background: 'rgba(255, 255, 255, 0.08)',
            padding: '2px 6px',
            borderRadius: '4px',
            fontFamily: 'monospace',
            fontSize: '12px',
            color: '#ff79c6'
          }}>
            {matchText.slice(1, -1)}
          </code>
        );
      } else if (matchText.startsWith('[') && matchText.includes('](')) {
        // Link markdown [text](url)
        const label = matchText.match(/\[(.*?)\]/)?.[1] || '';
        const url = matchText.match(/\((.*?)\)/)?.[1] || '';
        parts.push(
          <a key={matchStart} href={url} target="_blank" rel="noopener noreferrer" style={{
            color: 'var(--primary)',
            textDecoration: 'underline'
          }}>
            {label}
          </a>
        );
      }

      currentIdx = inlineRegex.lastIndex;
    }

    if (currentIdx < text.length) {
      parts.push(text.substring(currentIdx));
    }

    return parts.length > 0 ? parts : text;
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: 'rgba(0,0,0,0.2)',
      borderRadius: 'var(--radius-lg)',
      border: '1px solid var(--border-color)',
      overflow: 'hidden'
    }}>
      {/* Header status bar */}
      <div style={{
        padding: '12px 18px',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(255, 255, 255, 0.01)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: isConnected ? 'var(--accent)' : 'var(--danger)',
            boxShadow: isConnected ? '0 0 8px var(--accent)' : 'none'
          }} />
          <span style={{ fontWeight: 600, fontSize: '13px' }}>
            {isConnected ? 'Live Assistant Connection' : 'Assistant Offline'}
          </span>
        </div>
        {activeAgent && (
          <div style={{
            fontSize: '11px',
            background: 'var(--primary-glow)',
            color: 'var(--primary)',
            padding: '2px 8px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}>
            <span className="spinner" style={{ width: '10px', height: '10px', borderWidth: '1px' }} />
            {activeAgent}
          </div>
        )}
      </div>

      {/* Message feed */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        {messages.length === 0 && !streamingResponse && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--text-muted)',
            textAlign: 'center',
            padding: '40px'
          }}>
            <ShieldCheck size={48} style={{ color: 'var(--border-color-hover)', marginBottom: '16px' }} />
            <p style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-secondary)' }}>
              No messages in this workspace yet.
            </p>
            <p style={{ fontSize: '12px', marginTop: '6px' }}>
              Ask a research question or update the active step to begin experiment guidance.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
              width: '100%',
              maxWidth: '85%',
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start'
            }}
          >
            <div style={{
              background: msg.role === 'user' ? 'var(--primary)' : 'rgba(255, 255, 255, 0.03)',
              border: msg.role === 'user' ? 'none' : '1px solid var(--border-color)',
              color: msg.role === 'user' ? '#fff' : 'var(--text-primary)',
              borderRadius: msg.role === 'user' ? '18px 18px 2px 18px' : '18px 18px 18px 2px',
              padding: '12px 16px',
              boxShadow: 'var(--shadow-lg)'
            }}>
              {renderMarkdown(msg.content)}
            </div>
            
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginTop: '4px',
              padding: '0 4px',
              fontSize: '10px',
              color: 'var(--text-muted)'
            }}>
              <span>{msg.timestamp}</span>
              {msg.agentType && (
                <>
                  <span>•</span>
                  <span style={{ textTransform: 'capitalize', color: 'var(--primary)' }}>
                    {msg.agentType.replace('_', ' ')}
                  </span>
                </>
              )}
            </div>
          </div>
        ))}

        {streamingResponse && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-start',
              width: '100%',
              maxWidth: '85%',
              alignSelf: 'flex-start'
            }}
          >
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              borderRadius: '18px 18px 18px 2px',
              padding: '12px 16px',
              boxShadow: 'var(--shadow-lg)'
            }}>
              {renderMarkdown(streamingResponse)}
              <span className="spinner" style={{
                display: 'inline-block',
                width: '6px',
                height: '6px',
                borderWidth: '1px',
                marginLeft: '4px',
                verticalAlign: 'middle'
              }} />
            </div>
            <div style={{ marginTop: '4px', fontSize: '10px', color: 'var(--text-muted)' }}>
              Streaming Assistant response...
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSend} style={{
        padding: '16px',
        borderTop: '1px solid var(--border-color)',
        display: 'flex',
        gap: '12px',
        background: 'rgba(0, 0, 0, 0.1)'
      }}>
        <input
          className="glass-input"
          type="text"
          placeholder="Ask a scientific question or state your action..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={!isConnected}
          style={{ flex: 1, padding: '12px 16px' }}
        />
        <button
          className="btn-primary"
          type="submit"
          disabled={!isConnected || !input.trim()}
          style={{ width: '48px', height: '48px', padding: 0 }}
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};
