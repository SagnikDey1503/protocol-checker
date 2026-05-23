import { API_BASE_URL } from './api';

export interface WSHandlers {
  onToken: (token: string) => void;
  onDone: (data: { conversation_id: string; agent_type: string }) => void;
  onError: (error: string) => void;
  onClose: () => void;
  onOpen: () => void;
}

export class ChatWebSocket {
  private socket: WebSocket | null = null;
  private sessionId: string;
  private handlers: WSHandlers;
  private isManualClose = false;

  constructor(sessionId: string, handlers: WSHandlers) {
    this.sessionId = sessionId;
    this.handlers = handlers;
  }

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isManualClose = false;
    const token = localStorage.getItem('token');
    if (!token) {
      this.handlers.onError('No authentication token found. Please log in.');
      return;
    }

    const wsBase = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
    const wsProtocolBase = wsBase.replace(/^http/, 'ws');
    const wsUrl = `${wsProtocolBase}/ws/chat/${this.sessionId}?token=${encodeURIComponent(token)}`;
    
    try {
      this.socket = new WebSocket(wsUrl);
      
      this.socket.onopen = () => {
        this.handlers.onOpen();
      };
      
      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.token) {
            this.handlers.onToken(data.token);
          }
          
          if (data.done) {
            this.handlers.onDone({
              conversation_id: data.conversation_id,
              agent_type: data.agent_type
            });
          }
          
          if (data.error) {
            this.handlers.onError(data.token || 'WebSocket processing error');
          }
        } catch (err) {
          // If message is raw text token fallback
          this.handlers.onToken(event.data);
        }
      };
      
      this.socket.onerror = () => {
        if (!this.isManualClose) {
          this.handlers.onError('WebSocket connection error.');
        }
      };
      
      this.socket.onclose = (event) => {
        if (!this.isManualClose) {
          if (event.code === 1008) {
            this.handlers.onError('WebSocket authentication failed. Please log in again.');
          } else if (event.code !== 1000) {
            const reason = event.reason || 'Unexpected disconnect.';
            this.handlers.onError(`WebSocket closed (${event.code}). ${reason}`);
          }
        }
        this.handlers.onClose();
      };
    } catch (err: any) {
      this.handlers.onError(err.message || 'Failed to establish WebSocket connection.');
    }
  }

  sendMessage(message: string, experimentId?: string) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected.');
    }
    
    const payload = {
      message,
      experiment_id: experimentId || null
    };
    
    this.socket.send(JSON.stringify(payload));
  }

  disconnect() {
    if (this.socket) {
      this.isManualClose = true;
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
    }
  }
}
