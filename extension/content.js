// AI Hub 浏览器插件 - 内容脚本

let resultPanel = null;
let selectedText = '';

// 监听文本选择
document.addEventListener('mouseup', (e) => {
  const selection = window.getSelection();
  selectedText = selection.toString().trim();
  
  if (selectedText.length > 0 && selectedText.length < 2000) {
    showQuickButton(e.pageX, e.pageY);
  } else {
    hideQuickButton();
  }
});

// 显示快捷按钮
function showQuickButton(x, y) {
  hideQuickButton();
  
  const button = document.createElement('div');
  button.id = 'aihub-quick-btn';
  button.innerHTML = '🤖';
  button.style.cssText = `
    position: absolute;
    left: ${x}px;
    top: ${y - 40}px;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 2147483647;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    font-size: 16px;
    transition: transform 0.2s;
  `;
  
  button.addEventListener('mouseenter', () => {
    button.style.transform = 'scale(1.1)';
  });
  
  button.addEventListener('mouseleave', () => {
    button.style.transform = 'scale(1)';
  });
  
  button.addEventListener('click', (e) => {
    e.stopPropagation();
    showQuickMenu(x, y);
  });
  
  document.body.appendChild(button);
  
  // 3秒后自动隐藏
  setTimeout(hideQuickButton, 3000);
}

// 隐藏快捷按钮
function hideQuickButton() {
  const btn = document.getElementById('aihub-quick-btn');
  if (btn) btn.remove();
}

// 显示快捷菜单
function showQuickMenu(x, y) {
  hideQuickButton();
  hideQuickMenu();
  
  const menu = document.createElement('div');
  menu.id = 'aihub-quick-menu';
  menu.innerHTML = `
    <div class="aihub-menu-item" data-action="translate">🌐 翻译</div>
    <div class="aihub-menu-item" data-action="explain">💡 解释</div>
    <div class="aihub-menu-item" data-action="summarize">📝 摘要</div>
    <div class="aihub-menu-item" data-action="ask">❓ 提问</div>
  `;
  menu.style.cssText = `
    position: absolute;
    left: ${x}px;
    top: ${y}px;
    background: #1a1a2e;
    border-radius: 12px;
    padding: 8px;
    z-index: 2147483647;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  `;
  
  menu.querySelectorAll('.aihub-menu-item').forEach(item => {
    item.style.cssText = `
      padding: 10px 16px;
      color: #e8e8f0;
      cursor: pointer;
      border-radius: 8px;
      font-size: 14px;
      transition: background 0.2s;
    `;
    
    item.addEventListener('mouseenter', () => {
      item.style.background = '#252542';
    });
    
    item.addEventListener('mouseleave', () => {
      item.style.background = 'transparent';
    });
    
    item.addEventListener('click', () => {
      const action = item.dataset.action;
      processText(action, selectedText);
      hideQuickMenu();
    });
  });
  
  document.body.appendChild(menu);
  
  // 点击其他地方关闭
  setTimeout(() => {
    document.addEventListener('click', hideQuickMenu, { once: true });
  }, 0);
}

// 隐藏快捷菜单
function hideQuickMenu() {
  const menu = document.getElementById('aihub-quick-menu');
  if (menu) menu.remove();
}

// 处理文本
async function processText(action, text) {
  const { apiUrl, apiToken } = await chrome.storage.sync.get(['apiUrl', 'apiToken']);
  const baseUrl = apiUrl || 'http://localhost:8000';
  
  const prompts = {
    translate: `请将以下内容翻译成中文（如果是中文则翻译成英文）：\n\n${text}`,
    explain: `请用简单易懂的语言解释以下内容：\n\n${text}`,
    summarize: `请用2-3句话总结以下内容：\n\n${text}`,
    ask: `关于以下内容，请回答：\n\n${text}`
  };
  
  showResultPanel('处理中...', true);
  
  try {
    const response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiToken || ''}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompts[action] }],
        max_tokens: 500
      })
    });
    
    const data = await response.json();
    const result = data.choices?.[0]?.message?.content || '处理失败';
    showResultPanel(result, false);
  } catch (error) {
    showResultPanel(`错误: ${error.message}`, false);
  }
}

// 显示结果面板
function showResultPanel(content, loading = false) {
  hideResultPanel();
  
  resultPanel = document.createElement('div');
  resultPanel.id = 'aihub-result-panel';
  resultPanel.innerHTML = `
    <div class="aihub-panel-header">
      <span>🤖 AI Hub</span>
      <span class="aihub-close" onclick="document.getElementById('aihub-result-panel').remove()">×</span>
    </div>
    <div class="aihub-panel-content">
      ${loading ? '<div class="aihub-loading">处理中...</div>' : content}
    </div>
    <div class="aihub-panel-footer">
      <button class="aihub-btn" onclick="navigator.clipboard.writeText(this.parentElement.previousElementSibling.textContent)">📋 复制</button>
    </div>
  `;
  
  resultPanel.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 360px;
    max-height: 400px;
    background: #1a1a2e;
    border-radius: 16px;
    z-index: 2147483647;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    overflow: hidden;
  `;
  
  document.body.appendChild(resultPanel);
  
  // 添加样式
  const style = document.createElement('style');
  style.textContent = `
    .aihub-panel-header {
      padding: 16px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 600;
    }
    .aihub-close {
      cursor: pointer;
      font-size: 20px;
      opacity: 0.8;
    }
    .aihub-close:hover { opacity: 1; }
    .aihub-panel-content {
      padding: 16px;
      color: #e8e8f0;
      font-size: 14px;
      line-height: 1.6;
      max-height: 280px;
      overflow-y: auto;
    }
    .aihub-loading {
      text-align: center;
      color: #8888a0;
    }
    .aihub-panel-footer {
      padding: 12px 16px;
      border-top: 1px solid #2d2d4a;
    }
    .aihub-btn {
      padding: 8px 16px;
      background: #6366f1;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
    }
    .aihub-btn:hover { background: #5855eb; }
  `;
  resultPanel.appendChild(style);
}

// 隐藏结果面板
function hideResultPanel() {
  if (resultPanel) {
    resultPanel.remove();
    resultPanel = null;
  }
}

// 监听来自 background 的消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'showResult') {
    if (request.loading) {
      showResultPanel('处理中...', true);
    } else {
      showResultPanel(request.result, false);
    }
  }
  
  if (request.action === 'getPageContent') {
    const content = document.body.innerText.slice(0, 5000);
    sendResponse({ content });
  }
  
  return true;
});