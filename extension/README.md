# AI Hub 浏览器插件

AI Hub 的浏览器扩展，支持划词翻译、网页摘要、智能问答等功能。

## 功能特性

- 🌐 **划词翻译** - 选中文本后一键翻译
- 💡 **智能解释** - AI 解释选中的内容
- 📝 **快速摘要** - 总结选中文本或整个页面
- ❓ **智能问答** - 基于选中内容提问
- 🚀 **快捷对话** - 在 Popup 中直接与 AI 对话

## 安装方法

### Chrome / Edge

1. 打开浏览器扩展管理页面
   - Chrome: `chrome://extensions/`
   - Edge: `edge://extensions/`

2. 开启「开发者模式」

3. 点击「加载已解压的扩展程序」

4. 选择 `extension` 文件夹

### 图标生成

由于 Chrome 扩展需要 PNG 图标，请使用以下方法生成：

```bash
# 使用 ImageMagick 转换 SVG 到 PNG
convert icons/icon.svg -resize 16x16 icons/icon16.png
convert icons/icon.svg -resize 48x48 icons/icon48.png
convert icons/icon.svg -resize 128x128 icons/icon128.png
```

或者使用在线工具将 `icons/icon.svg` 转换为 PNG 格式。

## 配置

1. 点击扩展图标打开 Popup
2. 在设置中填写：
   - **API 地址**: AI Hub 服务地址（默认 `http://localhost:8000`）
   - **API Token**: 您的 API Token（可选）
3. 点击「保存设置」

## 使用方法

### 划词功能

1. 在任意网页选中文本
2. 点击出现的 🤖 按钮
3. 选择需要的功能（翻译/解释/摘要/提问）

### 右键菜单

1. 选中文本后右键
2. 选择「AI Hub」菜单中的功能

### Popup 对话

1. 点击扩展图标
2. 在对话框中输入问题
3. 按 Enter 或点击发送

## 权限说明

- `activeTab`: 访问当前标签页内容
- `storage`: 保存设置
- `contextMenus`: 添加右键菜单

## 开发

```bash
# 修改代码后，在扩展管理页面点击「重新加载」即可
```

## 许可证

MIT License