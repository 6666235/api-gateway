// AI Hub 浏览器插件 - 背景脚本

// 创建右键菜单
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'aihub-translate',
    title: '🌐 AI 翻译',
    contexts: ['selection']
  });
  
  chrome.contextMenus.create({
    id: 'aihub-explain',
    title: '💡 AI 解释',
    contexts: ['selection']
  });
  
  chrome.contextMenus.create({
    id: 'aihub-summarize',
    title: '📝 AI 摘要',
    contexts: ['selection']
  });
  
  chrome.contextMenus.create({
    id: 'aihub-ask',
    title: '❓ 问 AI',
    contexts: ['selection']
  });
  
  chrome.contextMenus.create({
    id: 'aihub-summarize-page',
    title: '📄 摘要整个页面',
    contexts: ['page']
  });
});

// 处理右键菜单点击
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const text = info.selectionText || '';
  
  if (info.menuItemId === 'aihub-summarize-page') {
    // 获取页面内容
    chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' }, async (response) => {
      if (response && response.content) {
        await processWithAI('summarize', response.content, tab.id);
      }
    });
    return;
  }
  
  if (!text) return;
  
  const actionMap = {
    'aihub-translate': 'translate',
    'aihub-explain': 'explain',
    'aihub-summarize': 'summarize',
    'aihub-ask': 'ask'
  };
  
  const action = actionMap[info.menuItemId];
  if (action) {
    await processWithAI(action, text, tab.id);
  }
});

// 调用 AI 处理
async function processWithAI(action, text, tabId) {
  const { apiUrl, apiToken } = await chrome.storage.sync.get(['apiUrl', 'apiToken']);
  const baseUrl = apiUrl || 'http://localhost:8000';
  
  const prompts = {
    translate: `请将以下内容翻译成中文（如果是中文则翻译成英文）：\n\n${text}`,
    explain: `请用简单易懂的语言解释以下内容：\n\n${text}`,
    summarize: `请用2-3句话总结以下内容的要点：\n\n${text.slice(0, 3000)}`,
    ask: `关于以下内容，请回答：\n\n${text}`
  };
  
  try {
    // 显示加载状态
    chrome.tabs.sendMessage(tabId, { 
      action: 'showResult', 
      loading: true,
      title: getActionTitle(action)
    });
    
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
    
    // 显示结果
    chrome.tabs.sendMessage(tabId, { 
      action: 'showResult', 
      result,
      title: getActionTitle(action)
    });
    
  } catch (error) {
    chrome.tabs.sendMessage(tabId, { 
      action: 'showResult', 
      result: `错误: ${error.message}`,
      title: '错误'
    });
  }
}

function getActionTitle(action) {
  const titles = {
    translate: '🌐 翻译结果',
    explain: '💡 解释',
    summarize: '📝 摘要',
    ask: '❓ AI 回答'
  };
  return titles[action] || 'AI Hub';
}

// 监听来自 popup 的消息
chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request.action === 'chat') {
    handleChat(request.message).then(sendResponse);
    return true;
  }
});

async function handleChat(message) {
  const { apiUrl, apiToken } = await chrome.storage.sync.get(['apiUrl', 'apiToken']);
  const baseUrl = apiUrl || 'http://localhost:8000';
  
  try {
    const response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiToken || ''}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: message }],
        max_tokens: 1000
      })
    });
    
    const data = await response.json();
    return { success: true, result: data.choices?.[0]?.message?.content || '' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}