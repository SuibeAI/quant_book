# README

这是一个采用 annotated 风格编写的 AI金融量化课程在线书籍项目，整体形式参考 Annotated Transformer，并使用 Jupyter Book / MyST 生成静态站点。

当前仓库包含两类主要内容：

- 教学型 Notebook：位于 books 目录，用于组织课程正文与模型讲解。
- 章节配套代码：位于各章节目录中，用于沉淀可复用的数据处理、模型与训练代码。

## 目录结构

```text
.
├── books/
│   └── DeepLOB/
│       ├── annotated_deeplob.ipynb
│       ├── deeplob_dataset.py
│       ├── deeplob_model.py
│       ├── prepare_data.py
│       └── train.py
├── .github/workflows/
│   └── pages.yml
├── .myst-templates/
│   └── book-theme/
├── book_generate.sh
├── book_start.sh
└── README.md
```

各目录用途如下：

- books：课程正文、教学型 Notebook 与章节配套代码。
- .github/workflows：GitHub Pages 自动构建与发布配置。
- .myst-templates：本地 MyST 站点模板，避免构建时在线下载主题。
- _build：站点构建输出目录，不纳入版本控制。

## 代码下载

```bash
git clone https://github.com/SuibeAI/quant_book.git
cd quant_book
```

## 环境要求

建议使用 Python 3.10 及以上版本。

安装构建依赖：

```bash
python3 -m pip install -U jupyter-book
```

如果你需要在 Notebook 中运行模型训练或数据处理，建议额外安装对应章节依赖，例如 PyTorch、NumPy、pandas、scikit-learn、matplotlib 等。

## 构建书籍

仓库根目录提供了统一构建脚本 [book_generate.sh](book_generate.sh)。默认会扫描 `books/` 下的 notebook，并生成一个聚合的 Jupyter Book 站点，HTML 输出到 `_build/books`。

直接构建：

```bash
bash book_generate.sh
```

指定输入目录与输出目录：

```bash
./book_generate.sh ./books ./_build/books
```

当前仓库已经包含本地 MyST 模板：

```text
.myst-templates/book-theme
```

构建脚本会优先使用项目内的 `.myst-templates/book-theme`，避免每次构建都访问 GitHub 下载主题。

构建完成后，聚合站点入口位于：

- `_build/books/index.html`

## 本地预览

开发时推荐直接使用 [book_start.sh](book_start.sh)，它会生成聚合 Jupyter Book 源目录，并交给 `jupyter-book start` 原生预览服务：

```bash
./book_start.sh
```

默认访问地址：

```text
http://127.0.0.1:8000
```

也可以用 `PORT` 指定端口：

```bash
PORT=8765 ./book_start.sh
```

如果只想预览已经构建好的静态文件，也可以启动一个静态文件服务器：

```bash
python3 -m http.server 8000 --directory _build/books
```

启动后可在浏览器中访问：

```text
http://localhost:8000
```

## GitHub Pages 访问

```text
https://suibeai.github.io/quant_book/
```

## Jupyter Notebook 本地访问

在仓库根目录启动 Jupyter：

```bash
jupyter notebook
```

或使用更现代的界面：

```bash
jupyter lab
```

默认访问地址（以终端实际输出为准）：

```text
http://localhost:8888
```

如果在远程机器上运行，可使用：

```bash
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
```
