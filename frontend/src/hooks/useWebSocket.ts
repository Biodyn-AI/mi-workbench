import { useCallback, useEffect, useRef, useState } from "react";

export interface WsMessage {
  type: string;
  data: unknown;
  timestamp: string;
}

interface UseWebSocketReturn {
  messages: WsMessage[];
  connected: boolean;
  send: (data: unknown) => void;
  clearMessages: () => void;
}

export function useWebSocket(runId: string | undefined): UseWebSocketReturn {
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const bufferRef = useRef<unknown[]>([]);

  const connect = useCallback(() => {
    if (!runId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/runs/${runId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // Flush buffer
      for (const msg of bufferRef.current) {
        ws.send(JSON.stringify(msg));
      }
      bufferRef.current = [];
    };

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        setMessages((prev) => [...prev, msg]);
      } catch {
        setMessages((prev) => [
          ...prev,
          { type: "raw", data: event.data, timestamp: new Date().toISOString() },
        ]);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // Auto-reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [runId]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      bufferRef.current.push(data);
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, connected, send, clearMessages };
}
