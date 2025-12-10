# AI Hub 桌面客户端

基于 Electron 的 AI Hub 桌面应用。

## 功能特性

- 🖥️ 原生桌面体验
- 🔔 系统托盘
- ⌨️ 全局快捷键 (Ctrl+Shift+A)
- 🌙 跟随系统主题
- 📌 窗口置顶
- 🚀 开机自启动

## 开发

```bash
# 安装依赖
npm install

# 启动开发模式
npm start

# 启动开发模式（带 DevTools）
npm start -- --dev
```

## 构建

```bash
# 构建所有平台
npm run build

# 仅构建 Windows
npm run build:win

# 仅构建 macOS
npm run build:mac

# 仅构建 Linux
npm run build:linux
```

## 配置

应用配置存储在：
- Windows: `%APPDATA%/ai-hub-desktop/config.json`
- macOS: `~/Library/Application Support/ai-hub-desktop/config.json`
- Linux: `~/.config/ai-hub-desktop/config.json`

### 配置项

```json
{
  "serverUrl": "http://localhost:8000",
  "theme": "system",
  "alwaysOnTop": false,
  "startMinimized": false,
  "closeToTray": true,
  "globalShortcut": "CommandOrControl+Shift+A"
}
```

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+Shift+A | 显示/隐藏窗口 |
| Ctrl+N | 新建对话 |
| Ctrl+, | 打开设置 |

## 图标

请在 `assets` 目录放置以下图标文件：
- `icon.png` - 256x256 PNG (Linux/托盘)
- `icon.ico` - Windows 图标
- `icon.icns` - macOS 图标

可以使用 [electron-icon-builder](https://github.com/nicholaslee119/electron-icon-builder) 生成：

```bash
npx electron-icon-builder --input=icon.png --output=assets
```