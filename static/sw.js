// AI Hub Service Worker - 增强版
const CACHE_NAME = 'ai-hub-v2';
const STATIC_CACHE = 'ai-hub-static-v2';
const DYNAMIC_CACHE = 'ai-hub-dynamic-v2';

// 静态资源（优先缓存）
const STATIC_ASSETS = [
  '/',
  '/static/index.html',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// API 路径（不缓存）
const API_PATHS = ['/api/', '/v1/', '/ws/'];

// 缓存策略配置
const CACHE_CONFIG = {
  maxAge: 24 * 60 * 60 * 1000, // 24小时
  maxEntries: 100
};

// 安装 Service Worker
self.addEventListener('install', event => {
  console.log('[SW] Installing...');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// 激活
self.addEventListener('activate', event => {
  console.log('[SW] Activating...');
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys
          .filter(k => k !== STATIC_CACHE && k !== DYNAMIC_CACHE)
          .map(k => {
            console.log('[SW] Removing old cache:', k);
            return caches.delete(k);
          })
      ))
      .then(() => self.clients.claim())
  );
});

// 拦截请求
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  // API 请求不缓存
  if (API_PATHS.some(path => url.pathname.includes(path))) {
    return;
  }
  
  // WebSocket 不处理
  if (url.protocol === 'ws:' || url.protocol === 'wss:') {
    return;
  }
  
  // 静态资源：缓存优先
  if (isStaticAsset(url.pathname)) {
    event.respondWith(cacheFirst(event.request));
    return;
  }
  
  // 其他请求：网络优先，失败时使用缓存
  event.respondWith(networkFirst(event.request));
});

// 缓存优先策略
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    // 后台更新缓存
    updateCache(request);
    return cached;
  }
  return fetchAndCache(request, STATIC_CACHE);
}

// 网络优先策略
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }
    // 返回离线页面
    return caches.match('/');
  }
}

// 获取并缓存
async function fetchAndCache(request, cacheName) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[SW] Fetch failed:', error);
    throw error;
  }
}

// 后台更新缓存
async function updateCache(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response);
    }
  } catch (error) {
    // 静默失败
  }
}

// 判断是否为静态资源
function isStaticAsset(pathname) {
  const staticExtensions = ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2'];
  return staticExtensions.some(ext => pathname.endsWith(ext)) || 
         pathname === '/' || 
         pathname.endsWith('.html');
}

// 推送通知
self.addEventListener('push', event => {
  const data = event.data ? event.data.json() : {};
  
  const options = {
    body: data.body || '您有新消息',
    icon: '/static/icon-192.png',
    badge: '/static/icon-192.png',
    vibrate: [100, 50, 100],
    data: {
      url: data.url || '/',
      timestamp: Date.now()
    },
    actions: data.actions || [
      { action: 'open', title: '查看' },
      { action: 'close', title: '关闭' }
    ],
    tag: data.tag || 'ai-hub-notification',
    renotify: true
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'AI Hub', options)
  );
});

// 点击通知
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  if (event.action === 'close') {
    return;
  }
  
  const url = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then(windowClients => {
        // 查找已打开的窗口
        for (const client of windowClients) {
          if (client.url.includes(self.location.origin) && 'focus' in client) {
            client.navigate(url);
            return client.focus();
          }
        }
        // 打开新窗口
        if (clients.openWindow) {
          return clients.openWindow(url);
        }
      })
  );
});

// 后台同步
self.addEventListener('sync', event => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'sync-messages') {
    event.waitUntil(syncMessages());
  }
});

// 同步消息
async function syncMessages() {
  try {
    // 从 IndexedDB 获取待同步的消息
    const db = await openDB();
    const messages = await db.getAll('pending-messages');
    
    for (const msg of messages) {
      try {
        await fetch('/api/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(msg)
        });
        await db.delete('pending-messages', msg.id);
      } catch (error) {
        console.error('[SW] Failed to sync message:', error);
      }
    }
  } catch (error) {
    console.error('[SW] Sync failed:', error);
  }
}

// 简单的 IndexedDB 封装
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ai-hub-sw', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      resolve({
        getAll: (store) => new Promise((res, rej) => {
          const tx = db.transaction(store, 'readonly');
          const req = tx.objectStore(store).getAll();
          req.onsuccess = () => res(req.result);
          req.onerror = () => rej(req.error);
        }),
        delete: (store, key) => new Promise((res, rej) => {
          const tx = db.transaction(store, 'readwrite');
          const req = tx.objectStore(store).delete(key);
          req.onsuccess = () => res();
          req.onerror = () => rej(req.error);
        })
      });
    };
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending-messages')) {
        db.createObjectStore('pending-messages', { keyPath: 'id' });
      }
    };
  });
}

// 定期清理缓存
self.addEventListener('message', event => {
  if (event.data === 'cleanup-cache') {
    event.waitUntil(cleanupCache());
  }
});

async function cleanupCache() {
  const cache = await caches.open(DYNAMIC_CACHE);
  const requests = await cache.keys();
  const now = Date.now();
  
  for (const request of requests) {
    const response = await cache.match(request);
    const dateHeader = response.headers.get('date');
    if (dateHeader) {
      const cacheTime = new Date(dateHeader).getTime();
      if (now - cacheTime > CACHE_CONFIG.maxAge) {
        await cache.delete(request);
      }
    }
  }
  
  // 限制缓存条目数
  if (requests.length > CACHE_CONFIG.maxEntries) {
    const toDelete = requests.slice(0, requests.length - CACHE_CONFIG.maxEntries);
    for (const request of toDelete) {
      await cache.delete(request);
    }
  }
}

console.log('[SW] Service Worker loaded');
