const { app, BrowserWindow, Menu, Tray, ipcMain, shell, globalShortcut, nativeTheme } = require('electron');
const path = require('path');
const Store = require('electron-store');

const store = new Store();
let mainWindow;
let tray;

// 默认配置
const defaultConfig = {
  serverUrl: 'http://localhost:8000',
  theme: 'system',
  alwaysOnTop: false,
  startMinimized: false,
  closeToTray: true,
  globalShortcut: 'CommandOrControl+Shift+A'
};

function createWindow() {
  const config = store.get('config', defaultConfig);
  
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: 'AI Hub',
    icon: path.join(__dirname, 'assets/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    frame: true,
    alwaysOnTop: config.alwaysOnTop,
    show: !config.startMinimized
  });

  // 加载服务器地址
  mainWindow.loadURL(config.serverUrl);

  // 开发模式打开 DevTools
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  // 窗口关闭处理
  mainWindow.on('close', (event) => {
    if (config.closeToTray && !app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });

  // 外部链接在浏览器打开
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

function createTray() {
  tray = new Tray(path.join(__dirname, 'assets/icon.png'));
  
  const contextMenu = Menu.buildFromTemplate([
    { label: '显示窗口', click: () => mainWindow.show() },
    { label: '隐藏窗口', click: () => mainWindow.hide() },
    { type: 'separator' },
    { label: '新建对话', click: () => {
      mainWindow.show();
      mainWindow.webContents.executeJavaScript('newChat()');
    }},
    { type: 'separator' },
    { label: '设置', click: () => {
      mainWindow.show();
      mainWindow.webContents.executeJavaScript('showPage("general")');
    }},
    { type: 'separator' },
    { label: '退出', click: () => {
      app.isQuitting = true;
      app.quit();
    }}
  ]);
  
  tray.setToolTip('AI Hub');
  tray.setContextMenu(contextMenu);
  
  tray.on('click', () => {
    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
    }
  });
}

function registerGlobalShortcuts() {
  const config = store.get('config', defaultConfig);
  
  // 注册全局快捷键
  globalShortcut.register(config.globalShortcut, () => {
    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

function createMenu() {
  const template = [
    {
      label: '文件',
      submenu: [
        { label: '新建对话', accelerator: 'CmdOrCtrl+N', click: () => {
          mainWindow.webContents.executeJavaScript('newChat()');
        }},
        { type: 'separator' },
        { label: '设置', accelerator: 'CmdOrCtrl+,', click: () => {
          mainWindow.webContents.executeJavaScript('showPage("general")');
        }},
        { type: 'separator' },
        { role: 'quit', label: '退出' }
      ]
    },
    {
      label: '编辑',
      submenu: [
        { role: 'undo', label: '撤销' },
        { role: 'redo', label: '重做' },
        { type: 'separator' },
        { role: 'cut', label: '剪切' },
        { role: 'copy', label: '复制' },
        { role: 'paste', label: '粘贴' },
        { role: 'selectAll', label: '全选' }
      ]
    },
    {
      label: '视图',
      submenu: [
        { role: 'reload', label: '刷新' },
        { role: 'forceReload', label: '强制刷新' },
        { type: 'separator' },
        { role: 'zoomIn', label: '放大' },
        { role: 'zoomOut', label: '缩小' },
        { role: 'resetZoom', label: '重置缩放' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: '全屏' }
      ]
    },
    {
      label: '窗口',
      submenu: [
        { label: '置顶', type: 'checkbox', checked: store.get('config.alwaysOnTop', false), click: (item) => {
          mainWindow.setAlwaysOnTop(item.checked);
          store.set('config.alwaysOnTop', item.checked);
        }},
        { type: 'separator' },
        { role: 'minimize', label: '最小化' },
        { role: 'close', label: '关闭' }
      ]
    },
    {
      label: '帮助',
      submenu: [
        { label: '文档', click: () => shell.openExternal('https://github.com/6666235/api-gateway') },
        { label: '报告问题', click: () => shell.openExternal('https://github.com/6666235/api-gateway/issues') },
        { type: 'separator' },
        { label: '关于', click: () => {
          const { dialog } = require('electron');
          dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: '关于 AI Hub',
            message: 'AI Hub Desktop',
            detail: '版本: 2.0.0\n企业级统一 AI 平台'
          });
        }}
      ]
    }
  ];
  
  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// IPC 通信
ipcMain.handle('get-config', () => store.get('config', defaultConfig));
ipcMain.handle('set-config', (_event, config) => store.set('config', config));
ipcMain.handle('get-theme', () => nativeTheme.shouldUseDarkColors ? 'dark' : 'light');

// 应用启动
app.whenReady().then(() => {
  createWindow();
  createTray();
  createMenu();
  registerGlobalShortcuts();
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// 应用退出
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

// 单实例锁定
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}