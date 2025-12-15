/**
 * WebSocket 连接管理器
 * 支持自动重连、心跳检测、消息队列
 */
class WebSocketManager {
  constructor(options = {}) {
    this.url = options.url || null;
    this.protocols = options.protocols || [];
    this.reconnectInterval = options.reconnectInterval || 3000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.heartbeatMessage = options.heartbeatMessage || JSON.stringify({ type: 'ping' });
    this.debug = options.debug || false;
    
    this.ws = null;
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
    this.messageQueue = [];
    this.isConnecting = false;
    this.manualClose = false;
    
    // 事件回调
    this.onOpen = options.onOpen || (() => {});
    this.onClose = options.onClose || (() => {});
    this.onError = options.onError || (() => {});
    this.onMessage = options.onMessage || (() => {});
    this.onReconnect = options.onReconnect || (() => {});
    this.onReconnectFailed = options.onReconnectFailed || (() => {});
  }
  
  log(...args) {
    if (this.debug) {
      console.log('[WebSocket]', ...args);
    }
  }
  
  /**
   * 连接 WebSocket
   */
  connect(url = null) {
    if (url) {
      this.url = url;
    }
    
    if (!this.url) {
      throw new Error('WebSocket URL is required');
    }
    
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      this.log('Already connected or connecting');
      return;
    }
    
    this.isConnecting = true;
    this.manualClose = false;
    
    try {
      this.ws = new WebSocket(this.url, this.protocols);
      this.setupEventHandlers();
    } catch (error) {
      this.log('Connection error:', error);
      this.handleReconnect();
    }
  }
  
  /**
   * 设置事件处理器
   */
  setupEventHandlers() {
    this.ws.onopen = (event) => {
      this.log('Connected');
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      
      // 发送队列中的消息
      this.flushMessageQueue();
      
      // 启动心跳
      this.startHeartbeat();
      
      this.onOpen(event);
    };
    
    this.ws.onclose = (event) => {
      this.log('Disconnected:', event.code, event.reason);
      this.isConnecting = false;
      this.stopHeartbeat();
      
      this.onClose(event);
      
      // 非手动关闭时尝试重连
      if (!this.manualClose && !event.wasClean) {
        this.handleReconnect();
      }
    };
    
    this.ws.onerror = (error) => {
      this.log('Error:', error);
      this.onError(error);
    };
    
    this.ws.onmessage = (event) => {
      // 处理心跳响应
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') {
          this.log('Heartbeat received');
          return;
        }
      } catch (e) {
        // 非 JSON 消息
      }
      
      this.onMessage(event);
    };
  }
  
  /**
   * 处理重连
   */
  handleReconnect() {
    if (this.manualClose) {
      return;
    }
    
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.log('Max reconnect attempts reached');
      this.onReconnectFailed();
      return;
    }
    
    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.min(this.reconnectAttempts, 5);
    
    this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.onReconnect(this.reconnectAttempts);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  /**
   * 启动心跳
   */
  startHeartbeat() {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.log('Sending heartbeat');
        this.ws.send(this.heartbeatMessage);
      }
    }, this.heartbeatInterval);
  }
  
  /**
   * 停止心跳
   */
  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
  
  /**
   * 发送消息
   */
  send(data) {
    const message = typeof data === 'string' ? data : JSON.stringify(data);
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message);
      return true;
    } else {
      // 连接未就绪，加入队列
      this.log('Connection not ready, queuing message');
      this.messageQueue.push(message);
      
      // 尝试重连
      if (!this.isConnecting && !this.manualClose) {
        this.connect();
      }
      
      return false;
    }
  }
  
  /**
   * 发送队列中的消息
   */
  flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(message);
      } else {
        // 放回队列
        this.messageQueue.unshift(message);
        break;
      }
    }
  }
  
  /**
   * 关闭连接
   */
  close(code = 1000, reason = 'Normal closure') {
    this.manualClose = true;
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.ws) {
      this.ws.close(code, reason);
      this.ws = null;
    }
    
    this.log('Connection closed manually');
  }
  
  /**
   * 获取连接状态
   */
  getState() {
    if (!this.ws) {
      return 'CLOSED';
    }
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'CONNECTING';
      case WebSocket.OPEN:
        return 'OPEN';
      case WebSocket.CLOSING:
        return 'CLOSING';
      case WebSocket.CLOSED:
        return 'CLOSED';
      default:
        return 'UNKNOWN';
    }
  }
  
  /**
   * 是否已连接
   */
  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
  
  /**
   * 重置重连计数
   */
  resetReconnectAttempts() {
    this.reconnectAttempts = 0;
  }
}

// 创建全局实例
window.WebSocketManager = WebSocketManager;

// 便捷函数：创建聊天 WebSocket
function createChatWebSocket(conversationId, options = {}) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${window.location.host}/ws/chat/${conversationId}`;
  
  return new WebSocketManager({
    url,
    debug: options.debug || false,
    reconnectInterval: 2000,
    maxReconnectAttempts: 15,
    heartbeatInterval: 25000,
    onOpen: options.onOpen || (() => console.log('Chat connected')),
    onClose: options.onClose || (() => console.log('Chat disconnected')),
    onError: options.onError || ((e) => console.error('Chat error:', e)),
    onMessage: options.onMessage || ((e) => console.log('Chat message:', e.data)),
    onReconnect: options.onReconnect || ((attempt) => {
      console.log(`Reconnecting... (${attempt})`);
      showToast && showToast(`正在重连... (${attempt})`, 'warning');
    }),
    onReconnectFailed: options.onReconnectFailed || (() => {
      console.error('Failed to reconnect');
      showToast && showToast('连接失败，请刷新页面', 'error');
    })
  });
}

// 便捷函数：创建协作 WebSocket
function createCollaborationWebSocket(sessionId, userId, username, options = {}) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${window.location.host}/ws/collaboration/${sessionId}?user_id=${userId}&username=${encodeURIComponent(username)}`;
  
  return new WebSocketManager({
    url,
    debug: options.debug || false,
    reconnectInterval: 2000,
    maxReconnectAttempts: 20,
    heartbeatInterval: 20000,
    ...options
  });
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WebSocketManager, createChatWebSocket, createCollaborationWebSocket };
}
