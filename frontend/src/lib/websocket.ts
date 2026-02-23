export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp?: string;
}

export interface WebSocketEventHandlers {
  connect?: () => void;
  disconnect?: () => void;
  message?: (message: MessageEvent) => void;
  error?: (error: Event) => void;
}

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number = 5000;
  private maxReconnectAttempts: number = 10;
  private reconnectAttempts: number = 0;
  private shouldReconnect: boolean = true;
  private eventHandlers: WebSocketEventHandlers = {};
  private subscriptions: Set<string> = new Set();
  private messageQueue: WebSocketMessage[] = [];
  private isConnected: boolean = false;

  constructor(url?: string) {
    // Use provided URL or construct from environment
    this.url = url || this.getDefaultUrl();
  }

  private getDefaultUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsPath = process.env.NEXT_PUBLIC_WS_PATH || '/ws';
    return `${protocol}//${host}${wsPath}`;
  }

  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.warn('WebSocket is already connected');
      return;
    }

    try {
      this.ws = new WebSocket(this.url);
      this.setupEventListeners();
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  private setupEventListeners(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.resubscribe();
      this.flushMessageQueue();
      this.eventHandlers.connect?.();
    };

    this.ws.onmessage = (event) => {
      this.eventHandlers.message?.(event);
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected', event.code, event.reason);
      this.isConnected = false;
      this.ws = null;
      
      if (this.shouldReconnect) {
        this.scheduleReconnect();
      }
      
      this.eventHandlers.disconnect?.();
    };

    this.ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      this.eventHandlers.error?.(event);
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    console.log(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (this.shouldReconnect) {
        this.connect();
      }
    }, delay);
  }

  private resubscribe(): void {
    if (!this.isConnected) return;

    this.subscriptions.forEach(channel => {
      this.send({
        type: 'subscribe',
        payload: { channel }
      });
    });
  }

  private flushMessageQueue(): void {
    if (!this.isConnected) return;

    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  public disconnect(): void {
    console.log('Disconnecting WebSocket');
    this.shouldReconnect = false;
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.isConnected = false;
    this.messageQueue = [];
  }

  public send(message: WebSocketMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      // Queue message for sending when connected
      this.messageQueue.push(message);
      return;
    }

    try {
      const messageStr = JSON.stringify(message);
      this.ws.send(messageStr);
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      this.messageQueue.push(message);
    }
  }

  public subscribe(channel: string): void {
    this.subscriptions.add(channel);
    
    if (this.isConnected) {
      this.send({
        type: 'subscribe',
        payload: { channel }
      });
    }
  }

  public unsubscribe(channel: string): void {
    this.subscriptions.delete(channel);
    
    if (this.isConnected) {
      this.send({
        type: 'unsubscribe',
        payload: { channel }
      });
    }
  }

  public on(event: keyof WebSocketEventHandlers, handler: Function): void {
    this.eventHandlers[event] = handler as any;
  }

  public off(event: keyof WebSocketEventHandlers): void {
    delete this.eventHandlers[event];
  }

  public getConnectionState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  public isConnectionOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  public getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  public getSubscriptions(): string[] {
    return Array.from(this.subscriptions);
  }

  // Static method to create a singleton instance
  public static createInstance(url?: string): WebSocketManager {
    return new WebSocketManager(url);
  }
}

// Export a singleton instance for global use
export const wsManager = WebSocketManager.createInstance();

// Hook for React components
export const useWebSocket = (url?: string) => {
  const [wsManager, setWsManager] = useState<WebSocketManager | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<number>(WebSocket.CLOSED);

  useEffect(() => {
    const manager = new WebSocketManager(url);
    setWsManager(manager);

    manager.on('connect', () => {
      setIsConnected(true);
      setConnectionState(WebSocket.OPEN);
    });

    manager.on('disconnect', () => {
      setIsConnected(false);
      setConnectionState(WebSocket.CLOSED);
    });

    manager.on('error', (error) => {
      console.error('WebSocket error in hook:', error);
    });

    manager.connect();

    return () => {
      manager.disconnect();
    };
  }, [url]);

  return {
    wsManager,
    isConnected,
    connectionState
  };
};

import { useState, useEffect } from 'react';

// Hook for subscribing to specific channels
export const useWebSocketSubscription = (channels: string[], url?: string) => {
  const { wsManager, isConnected } = useWebSocket(url);
  const [messages, setMessages] = useState<any[]>([]);

  useEffect(() => {
    if (!wsManager || !isConnected) return;

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        setMessages(prev => [...prev, data]);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    wsManager.on('message', handleMessage);

    // Subscribe to channels
    channels.forEach(channel => {
      wsManager.subscribe(channel);
    });

    return () => {
      wsManager.off('message');
      channels.forEach(channel => {
        wsManager.unsubscribe(channel);
      });
    };
  }, [wsManager, isConnected, channels]);

  return {
    messages,
    wsManager,
    isConnected
  };
};