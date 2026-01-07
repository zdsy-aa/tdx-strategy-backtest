# 网站代码目录 (web/)

> 包含 TradeGuide 交易指标参考系统的完整网站代码。

## 📁 目录结构

```
web/
├── README.md                # 本说明文件
├── client/                  # 前端代码
│   ├── src/
│   │   ├── pages/           # 页面组件
│   │   ├── components/      # 可复用组件
│   │   ├── data/            # 静态数据
│   │   ├── lib/             # 工具函数
│   │   └── contexts/        # React 上下文
│   ├── public/              # 静态资源
│   └── index.html           # HTML 入口
├── server/                  # 后端代码
│   ├── routers.ts           # API 路由定义
│   ├── _core/               # 核心模块
│   └── index.ts             # 服务器入口
└── shared/                  # 前后端共享代码
    └── const.ts             # 共享常量
```

## 🛠️ 技术栈

### 前端
- **框架**：React 19
- **样式**：Tailwind CSS 4 + shadcn/ui
- **路由**：Wouter
- **状态管理**：React Context + tRPC
- **图表**：Recharts

### 后端
- **运行时**：Node.js
- **框架**：Express
- **API**：tRPC
- **数据库**：PostgreSQL + Drizzle ORM

## 📄 页面说明

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页 | `/` | 系统概览和核心策略展示 |
| 指标详解 | `/indicators` | 技术指标的详细说明和K线案例 |
| 组合方案 | `/strategies` | 不同风险偏好的策略推荐 |
| 回测数据 | `/backtest` | 历史回测结果可视化 |
| AI 优化 | `/ai-optimizer` | 自然语言策略优化 |
| 自定义策略 | `/custom-strategy` | 自由组合指标并回测 |
| 回测报告 | `/reports` | 查看详细回测报告 |

## 🚀 本地开发

### 1. 安装依赖
```bash
cd web
pnpm install
```

### 2. 启动开发服务器
```bash
pnpm dev
```
访问地址通常为 `http://localhost:3000`。

### 3. 构建生产版本
```bash
pnpm build
```

### 4. 启动生产服务器
```bash
pnpm start
```

## 📊 数据关联

网页展示的数据主要来自 `../data/backtest_results/` 目录下的 JSON 文件：
*   `backtest_results.json`: 用于仪表盘和策略详情页。
*   `stock_reports.json`: 用于股票明细页。

请确保在运行网页前，已通过 `py_file/full_backtest.py` 生成了最新的回测数据。

## 🔧 配置说明

### 环境变量
```env
# 数据库
DATABASE_URL=postgresql://...

# AI 服务
BUILT_IN_FORGE_API_KEY=...
BUILT_IN_FORGE_API_URL=...

# OAuth
OAUTH_SERVER_URL=...
JWT_SECRET=...
```

### 端口配置
- 开发服务器：3000
- 生产服务器：3000（可通过 PORT 环境变量修改）

## 📦 组件库

项目使用 shadcn/ui 组件库，主要组件包括：

- `Button` - 按钮
- `Card` - 卡片
- `Dialog` - 对话框
- `Tabs` - 标签页
- `Badge` - 徽章
- `Select` - 下拉选择
- `Checkbox` - 复选框
- `Slider` - 滑块

## 🎨 设计系统

### 主题
- 默认深色主题
- Glassmorphism 风格
- 金融科技配色

### 颜色变量
```css
--primary: 蓝紫渐变
--background: #0a0a0a
--foreground: #fafafa
--accent: 青色/紫色
```

## ⚠️ 注意事项

1. **数据同步**：确保 `data/` 目录有最新数据
2. **API 依赖**：部分功能需要后端 API 支持
3. **浏览器兼容**：建议使用 Chrome/Edge 最新版本
4. **移动端**：已适配移动端，但部分图表在小屏幕上显示效果有限
