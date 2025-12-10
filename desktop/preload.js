const { contextBridge, ipcRenderer } = require('electron');

// 暴露安全的 API 给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
  // 配置
  getConfig: () => ipcRenderer.invoke('get-config'),
  setConfig: (config) => ipcRenderer.invoke('set-config', config),
  
  // 主题
  getTheme: () => ipcRenderer.invoke('get-theme'),
  
  // 平台信息
  platform: process.platform,
  
  // 版本
  version: '2.0.0'
});