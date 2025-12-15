/**
 * AI Hub 增强功能
 * 包含：快捷键、主题切换、性能优化、智能建议等
 */

// ========== 快捷键管理 ==========
const KeyboardShortcuts = {
    shortcuts: {},
    
    init() {
        document.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.registerDefaults();
    },
    
    registerDefaults() {
        this.register('Ctrl+N', () => { if(typeof newChat === 'function') newChat(); });
        this.register('Ctrl+L', () => { if(typeof clearChat === 'function' && confirm('清空对话？')) clearChat(); });
        this.register('Ctrl+K', () => this.showSearchDialog());
        this.register('Ctrl+B', () => this.toggleSidebar());
        this.register('Ctrl+,', () => this.showSettings());
        this.register('Ctrl+E', () => { if(typeof exportChat === 'function') exportChat('md'); });
        this.register('Escape', () => this.closeModals());
    },
    
    register(shortcut, callback) {
        this.shortcuts[shortcut.toLowerCase()] = callback;
    },
    
    handleKeydown(e) {
        const key = this.getKeyString(e);
        const callback = this.shortcuts[key];
        if (callback && !this.isInputFocused()) {
            e.preventDefault();
            callback();
        }
    },
    
    getKeyString(e) {
        const parts = [];
        if (e.ctrlKey || e.metaKey) parts.push('ctrl');
        if (e.shiftKey) parts.push('shift');
        if (e.altKey) parts.push('alt');
        parts.push(e.key.toLowerCase());
        return parts.join('+');
    },
    
    isInputFocused() {
        const active = document.activeElement;
        return active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA');
    },
    
    showSearchDialog() {
        const existing = document.getElementById('searchDialog');
        if (existing) { existing.remove(); return; }
        
        const dialog = document.createElement('div');
        dialog.id = 'searchDialog';
        dialog.innerHTML = `
            <div style="position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:1000;display:flex;align-items:flex-start;justify-content:center;padding-top:100px">
                <div style="background:var(--bg2);border-radius:12px;width:500px;max-width:90%;box-shadow:0 20px 60px rgba(0,0,0,0.3)">
                    <input type="text" id="searchInput" placeholder="搜索对话、命令..." 
                        style="width:100%;padding:16px 20px;border:none;background:transparent;color:var(--text);font-size:16px;outline:none">
                    <div id="searchResults" style="max-height:300px;overflow-y:auto;border-top:1px solid var(--border)"></div>
                </div>
            </div>
        `;
        dialog.onclick = (e) => { if(e.target === dialog.firstElementChild) dialog.remove(); };
        document.body.appendChild(dialog);
        document.getElementById('searchInput').focus();
        
        document.getElementById('searchInput').oninput = (e) => {
            const query = e.target.value.toLowerCase();
            this.performSearch(query);
        };
    },
    
    performSearch(query) {
        const results = document.getElementById('searchResults');
        if (!query) { results.innerHTML = ''; return; }
        
        let html = '';
        // 搜索对话
        if (typeof chats !== 'undefined') {
            const matches = chats.filter(c => 
                (c.title && c.title.toLowerCase().includes(query)) ||
                (c.messages && c.messages.some(m => m.content && m.content.toLowerCase().includes(query)))
            ).slice(0, 5);
            
            matches.forEach(c => {
                html += `<div class="search-result" onclick="selectChat('${c.id}');document.getElementById('searchDialog').remove()" 
                    style="padding:12px 20px;cursor:pointer;border-bottom:1px solid var(--border)">
                    <div style="font-weight:500">${c.title || '新对话'}</div>
                    <div style="font-size:12px;color:var(--text2)">${c.messages ? c.messages.length : 0} 条消息</div>
                </div>`;
            });
        }
        
        // 命令建议
        const commands = [
            {cmd: '/help', desc: '显示帮助'},
            {cmd: '/clear', desc: '清空对话'},
            {cmd: '/export', desc: '导出对话'},
            {cmd: '/new', desc: '新建对话'}
        ].filter(c => c.cmd.includes(query) || c.desc.includes(query));
        
        commands.forEach(c => {
            html += `<div class="search-result" onclick="document.getElementById('chatInput').value='${c.cmd}';document.getElementById('searchDialog').remove()" 
                style="padding:12px 20px;cursor:pointer;border-bottom:1px solid var(--border)">
                <div style="font-weight:500">${c.cmd}</div>
                <div style="font-size:12px;color:var(--text2)">${c.desc}</div>
            </div>`;
        });
        
        results.innerHTML = html || '<div style="padding:20px;text-align:center;color:var(--text2)">无结果</div>';
    },
    
    toggleSidebar() {
        const nav = document.querySelector('.nav');
        if (nav) {
            nav.style.display = nav.style.display === 'none' ? 'flex' : 'none';
        }
    },
    
    showSettings() {
        const settingsNav = document.querySelector('.nav-item[data-page="settings"]');
        if (settingsNav) settingsNav.click();
    },
    
    closeModals() {
        document.querySelectorAll('#searchDialog, .modal-overlay').forEach(el => el.remove());
    }
};


// ========== 主题管理 ==========
const ThemeManager = {
    themes: {
        dark: { name: '深色', icon: '🌙' },
        light: { name: '浅色', icon: '☀️' },
        blue: { name: '蓝色', icon: '💙' },
        green: { name: '绿色', icon: '💚' },
        purple: { name: '紫色', icon: '💜' }
    },
    
    init() {
        const saved = localStorage.getItem('theme') || 'dark';
        this.apply(saved);
    },
    
    apply(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    },
    
    toggle() {
        const current = localStorage.getItem('theme') || 'dark';
        const themes = Object.keys(this.themes);
        const idx = themes.indexOf(current);
        const next = themes[(idx + 1) % themes.length];
        this.apply(next);
        return next;
    },
    
    getSelector() {
        let html = '<div style="display:flex;gap:8px;flex-wrap:wrap">';
        for (const [id, theme] of Object.entries(this.themes)) {
            const active = localStorage.getItem('theme') === id ? 'border:2px solid var(--accent)' : '';
            html += `<button onclick="ThemeManager.apply('${id}')" 
                style="padding:8px 16px;border-radius:8px;background:var(--bg3);border:2px solid transparent;cursor:pointer;${active}">
                ${theme.icon} ${theme.name}
            </button>`;
        }
        html += '</div>';
        return html;
    }
};


// ========== 性能监控 ==========
const PerformanceTracker = {
    metrics: [],
    
    record(provider, model, latency, tokens, success) {
        this.metrics.push({
            timestamp: Date.now(),
            provider, model, latency, tokens, success
        });
        // 只保留最近 100 条
        if (this.metrics.length > 100) {
            this.metrics = this.metrics.slice(-100);
        }
        this.updateDisplay();
    },
    
    getStats() {
        if (this.metrics.length === 0) return null;
        
        const recent = this.metrics.filter(m => Date.now() - m.timestamp < 3600000);
        if (recent.length === 0) return null;
        
        const latencies = recent.map(m => m.latency);
        const successCount = recent.filter(m => m.success).length;
        
        return {
            calls: recent.length,
            avgLatency: (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(2),
            minLatency: Math.min(...latencies).toFixed(2),
            maxLatency: Math.max(...latencies).toFixed(2),
            successRate: ((successCount / recent.length) * 100).toFixed(1),
            totalTokens: recent.reduce((a, m) => a + (m.tokens || 0), 0)
        };
    },
    
    updateDisplay() {
        const stats = this.getStats();
        if (!stats) return;
        
        const el = document.getElementById('perfStats');
        if (el) {
            el.innerHTML = `
                <span title="平均响应时间">⚡ ${stats.avgLatency}s</span>
                <span title="成功率">✅ ${stats.successRate}%</span>
                <span title="总 Token">🔢 ${stats.totalTokens}</span>
            `;
        }
    }
};


// ========== 消息增强 ==========
const MessageEnhancer = {
    // 代码复制按钮
    addCopyButtons() {
        document.querySelectorAll('pre code').forEach(block => {
            if (block.parentElement.querySelector('.copy-btn')) return;
            
            const btn = document.createElement('button');
            btn.className = 'copy-btn';
            btn.innerHTML = '📋';
            btn.title = '复制代码';
            btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 8px;background:var(--bg3);border:none;border-radius:4px;cursor:pointer;opacity:0;transition:opacity 0.2s';
            
            btn.onclick = () => {
                navigator.clipboard.writeText(block.textContent);
                btn.innerHTML = '✅';
                setTimeout(() => btn.innerHTML = '📋', 2000);
            };
            
            block.parentElement.style.position = 'relative';
            block.parentElement.appendChild(btn);
            
            block.parentElement.onmouseenter = () => btn.style.opacity = '1';
            block.parentElement.onmouseleave = () => btn.style.opacity = '0';
        });
    },
    
    // 图片预览
    enableImagePreview() {
        document.querySelectorAll('.message img').forEach(img => {
            if (img.dataset.previewEnabled) return;
            img.dataset.previewEnabled = 'true';
            img.style.cursor = 'pointer';
            img.onclick = () => {
                const overlay = document.createElement('div');
                overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.9);z-index:2000;display:flex;align-items:center;justify-content:center;cursor:pointer';
                overlay.innerHTML = `<img src="${img.src}" style="max-width:90%;max-height:90%;border-radius:8px">`;
                overlay.onclick = () => overlay.remove();
                document.body.appendChild(overlay);
            };
        });
    },
    
    // 链接处理
    processLinks() {
        document.querySelectorAll('.message a').forEach(link => {
            if (link.dataset.processed) return;
            link.dataset.processed = 'true';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
        });
    },
    
    // 运行所有增强
    enhance() {
        this.addCopyButtons();
        this.enableImagePreview();
        this.processLinks();
    }
};


// ========== 输入增强 ==========
const InputEnhancer = {
    init() {
        const input = document.getElementById('chatInput');
        if (!input) return;
        
        // 自动调整高度
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
        });
        
        // 历史记录
        this.history = [];
        this.historyIndex = -1;
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp' && !input.value) {
                e.preventDefault();
                this.navigateHistory(-1, input);
            } else if (e.key === 'ArrowDown' && this.historyIndex >= 0) {
                e.preventDefault();
                this.navigateHistory(1, input);
            }
        });
    },
    
    addToHistory(text) {
        if (text && text.trim()) {
            this.history.unshift(text);
            if (this.history.length > 50) this.history.pop();
            this.historyIndex = -1;
        }
    },
    
    navigateHistory(direction, input) {
        const newIndex = this.historyIndex + direction;
        if (newIndex >= -1 && newIndex < this.history.length) {
            this.historyIndex = newIndex;
            input.value = newIndex === -1 ? '' : this.history[newIndex];
        }
    }
};


// ========== 通知系统 ==========
const Notifications = {
    container: null,
    
    init() {
        this.container = document.createElement('div');
        this.container.id = 'notifications';
        this.container.style.cssText = 'position:fixed;top:20px;right:20px;z-index:3000;display:flex;flex-direction:column;gap:8px';
        document.body.appendChild(this.container);
    },
    
    show(message, type = 'info', duration = 3000) {
        const colors = {
            info: 'var(--accent)',
            success: 'var(--success)',
            warning: 'var(--warning)',
            error: 'var(--error)'
        };
        
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        
        const toast = document.createElement('div');
        toast.style.cssText = `
            padding:12px 20px;background:var(--bg2);border-left:4px solid ${colors[type]};
            border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.3);
            display:flex;align-items:center;gap:10px;animation:slideIn 0.3s ease;
        `;
        toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
        
        this.container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
};


// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    KeyboardShortcuts.init();
    ThemeManager.init();
    InputEnhancer.init();
    Notifications.init();
    
    // 监听消息渲染
    const observer = new MutationObserver(() => {
        MessageEnhancer.enhance();
    });
    
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        observer.observe(chatMessages, { childList: true, subtree: true });
    }
    
    console.log('🚀 AI Hub 增强功能已加载');
});


// 添加动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    .search-result:hover { background: var(--bg3); }
`;
document.head.appendChild(style);
