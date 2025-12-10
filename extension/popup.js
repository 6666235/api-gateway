// AI Hub 浏览器插件 - Popup 脚本

document.addEventListener('DOMContentLoaded', async () => {
  // 加载保存的设置
  const { apiUrl, apiToken } = await chrome.storage.sync.get(['apiUrl', 'apiToken']);
  if (apiUrl) document.getElementById('apiUrl').value = apiUrl;
  if (apiToken) document.getElementById('apiToken').value = apiToken;
});

// 保存设置
async function saveSettings() {
  const apiUrl = document.getElementById('apiUrl').value.trim() || 'http://localhost:8000';
  const apiToken = document.getElementById('apiToken').value.trim();
  
  await chrome.storage.sync.set({ apiUrl, apiToken });
  showStatus('设置已保存', 'success');
  
  // 测试连接
  try {
    const response = await fetch(`${apiUrl}/health`);
    if (response.ok) {
      showStatus('连接成功！', 'success');
    } else {
      showStatus('连接失败，请检查地址', 'error');
    }
  } catch (e) {
    showStatus('无法连接到服务器', 'error');
  }
}

// 显示状态
function showStatus(message, type) {
  const status = document.getElementById('status');
  status.textContent = message;
  status.className = `status ${type}`;
  status.style.display = 'block';
  
  setTimeout(() => {
    status.style.display = 'none';
  }, 3000);
}

// 发送聊天消息
async function sendChat() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;
  
  input.value = '';
  
  // 添加用户消息
  addMessage(message, 'user');
  
  // 发送到 AI
  const response = await chrome.runtime.sendMessage({ action: 'chat', message });
  
  if (response.success) {
    addMessage(response.result, 'assistant');
  } else {
    addMessage(`错误: ${response.error}`, 'assistant');
  }
}

// 添加消息到聊天框
function addMessage(text, role) {
  const chatBox = document.getElementById('chatBox');
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// 快捷操作
async function quickAction(action) {
  // 获取当前标签页选中的文本
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
    if (response && response.text) {
      // 发送到 background 处理
      chrome.runtime.sendMessage({ 
        action: 'process', 
        type: action, 
        text: response.text 
      });
    } else {
      showStatus('请先选择文本', 'error');
    }
  });
}

// 打开 AI Hub
function openHub() {
  chrome.tabs.create({ url: 'http://localhost:8000' });
}