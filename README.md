# 🚇 Xi'an Metro Assistant | 西安地铁出行小助手

[English](#english) | [中文](#中文)

---

## ✨ Project Highlights | 项目亮点

| Feature | Description |
|---------|-------------|
| 🤖 **Deep Learning NLP** | CNN-based text classification model for natural language station recognition |
| 🗺️ **BFS Route Planning** | Breadth-First Search algorithm for optimal path finding |
| 🔄 **Multi-line Support** | Comprehensive support for all Xi'an Metro lines (1-6, 8, 9, 14) |
| 🌐 **Bilingual Interface** | Full Chinese/English language support |

---

## English

### About | 项目简介

**Xi'an Metro Assistant** is an intelligent route planning system for Xi'an Metro, featuring:

- **Natural Language Processing**: CNN model trained on 500+ query samples for understanding user intent
- **Smart Route Planning**: BFS-based algorithm finds optimal routes including direct and transfer options
- **Modern Web Interface**: Clean, responsive UI with real-time route display

### Tech Stack | 技术栈

```
Frontend:    HTML5 + CSS3 + JavaScript
Backend:     Flask (Python)
ML Framework: TensorFlow/Keras (CNN)
Algorithm:    Breadth-First Search (BFS)
Data:        9 Metro Lines, 200+ Stations
```

### Project Structure | 项目结构

```
├── app.py                    # Flask web application
├── train_model.py            # CNN model training script
├── generate_dataset.py       # Dataset generation script
├── templates/
│   └── index.html            # Frontend interface
├── metro_cnn_model.keras     # Trained NLP model
├── metro_tokenizer.pkl       # Tokenizer for text processing
├── metro_stations.pkl       # Station name database
└── requirements.txt          # Python dependencies
```

### Setup & Running | 安装运行

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate dataset
python generate_dataset.py

# 3. (Optional) Retrain model
python train_model.py

# 4. Run the application
python app.py

# 5. Open browser
# Visit http://localhost:5000
```

### Core Algorithm: BFS Route Planning | 核心算法：BFS路径规划

The route finding uses **Breadth-First Search** to explore metro connections:

```python
def find_direct_routes(origin, destination):
    # Find routes on the same metro line
    # Check station indices to determine direction

def find_one_transfer_routes(origin, destination):
    # Find routes requiring exactly one transfer
    # Explore common stations between lines
```

**BFS Characteristics**:
- **Time Complexity**: O(V + E) where V = stations, E = connections
- **Space Complexity**: O(V) for queue storage
- **Guarantees**: Shortest path in unweighted graph

### NLP Model: CNN Text Classification | NLP模型：CNN文本分类

```
Input: Natural language query (e.g., "从西安北站到钟楼怎么走")
       ↓
Tokenization & Padding (max_len=20)
       ↓
Embedding Layer (128 dimensions)
       ↓
Conv1D (64 filters, kernel_size=3)
       ↓
GlobalMaxPooling
       ↓
Dense (64) → Dropout (0.3)
       ↓
Output: Start station probability distribution
```

**Model Performance**:
- Training samples: 500
- Test accuracy: >95%
- Early stopping prevents overfitting

### Metro Lines Supported | 支持的地铁线路

| Line | Color | Route | Type |
|------|-------|-------|------|
| Line 1 | 🔴 Red | Fangzhicheng → Xianxixizhan | Linear |
| Line 2 | 🟢 Green | Xi'an North Station → Changninggong | Linear |
| Line 3 | 🟣 Purple | Yuhuazhai → Baoshuiqu | Linear |
| Line 4 | 🔵 Blue | Xi'an North Station → Hangtianxincheng | Linear |
| Line 5 | 🟠 Orange | Chuangxinguandong → Matengkong | Linear |
| Line 6 | 🩷 Pink | Int'l Medical Center → Fangzhicheng | Linear |
| Line 8 | 🔵 Cyan | **Loop Line (39 stations)** | Circular |
| Line 9 | 🟤 Brown | Fangzhicheng → Qinlingxi | Linear |
| Line 14 | 💜 Purple | Airport → Heshao | Linear |

---

## 中文

### 项目简介

西安地铁出行小助手是一个智能地铁线路规划系统，具有以下特点：

- **自然语言处理**：基于CNN的文本分类模型，理解用户出行意图
- **智能路径规划**：基于BFS算法计算最优路线，支持直达和换乘方案
- **现代化Web界面**：简洁美观的响应式设计

### 技术架构

```
前端：      HTML5 + CSS3 + JavaScript (Ajax异步交互)
后端：      Flask (Python Web框架)
机器学习：   TensorFlow/Keras (CNN卷积神经网络)
核心算法：   广度优先搜索 (BFS)
数据：      9条地铁线路，200+站点
```

### 核心算法详解

#### 1. 直达路线查找 (BFS单线搜索)

```python
def find_direct_routes(origin, destination):
    # 同一线路站点间直接计算
    # 根据站点索引判断行进方向
    # 时间复杂度: O(n)
```

#### 2. 换乘路线查找 (BFS跨线搜索)

```python
def find_one_transfer_routes(origin, destination):
    # 遍历起点站所在的所有线路
    # 查找与目标线路的交集站点（换乘站）
    # 分别计算两段路程
```

#### 3. 时间估算

```
总时间 = 站点数 × 3分钟 + 换乘次数 × 5分钟
```

### NLP模型训练流程

```
Step 1: 数据生成 (generate_dataset.py)
        ↓ 500条query文本模板
Step 2: 文本序列化 (Tokenizer)
        ↓
Step 3: CNN模型训练 (train_model.py)
        - Embedding: 128维词向量
        - Conv1D: 64个卷积核
        - Dropout: 0.3防过拟合
        - EarlyStopping: patience=3
        ↓
Step 4: 模型保存
        - metro_cnn_model.keras
        - metro_tokenizer.pkl
        - metro_stations.pkl
```

### 功能演示

**支持的自然语言查询**：
- "从西安北站到钟楼怎么走"
- "我要从大雁塔去机场"
- "从小寨到工大怎么去"

**支持的交互方式**：
- 🔘 下拉菜单选择起点终点
- 💬 自然语言语音输入
- 🔄 一键交换起终点

### 运行截图

```
┌─────────────────────────────────────┐
│      🚇 西安地铁出行小助手           │
├─────────────────────────────────────┤
│  💬 自然语言查询                    │
│  [我要从东长安街去造字台]  [解析]   │
├─────────────────────────────────────┤
│  🚩 起点站    [东长安街 ▼]          │
│              ⇄                     │
│  🏁 终点站    [造字台 ▼]            │
│                                     │
│  [       查询路线       ]           │
├─────────────────────────────────────┤
│  📍 出行方案                        │
│  ┌─────────────────────────────┐   │
│  │ 🚇 4号线 → 6号线             │   │
│  │ 东长安街 → ... → 造字台       │   │
│  │ ⏱️ 约35分钟 | 📍 28站        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

---

## 📄 License | 许可证

MIT License - Feel free to use and modify!

---

## 👤 Author | 作者

**cheng-jun-hao**
Xi'an Metro Assistant - 创新开发项目

---

<p align="center">
  <strong>Made with ❤️ for Xi'an Metro</strong>
</p>
