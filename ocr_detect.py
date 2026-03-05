import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io
import time
import os
import csv

# ========== 从环境变量读取阿里云 DashScope API Key ==========
# 建议与项目中其它模块统一，使用 DASHSCOPE_API_KEY
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
# ========================================================

# 页面配置
st.set_page_config(
    page_title="千问OCR识别工具",
    page_icon="🤖",
    layout="wide"
)

# 初始化session state
if 'qwen_results' not in st.session_state:
    st.session_state.qwen_results = {}


def encode_image_bytes(image_bytes: bytes) -> str:
    """将图片字节转换为base64编码字符串"""
    return base64.b64encode(image_bytes).decode('utf-8')


def process_with_qwen(client, image_bytes: bytes, custom_prompt: str | None = None):
    """使用千问模型识别图片。

    参数 image_bytes 为原始图片二进制内容，方便支持本地上传或 zip 中的文件。
    """
    try:
        # 获取图片base64
        base64_image = encode_image_bytes(image_bytes)

        # 默认提示词
        if not custom_prompt:
            custom_prompt = "请提取这张图片中的所有文字内容，保持原始排版。只输出提取到的文字，不要添加任何解释。"

        # 调用千问API
        completion = client.chat.completions.create(
            model="qwen-vl-ocr-2025-11-20",  # 千问OCR专用模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": custom_prompt
                        }
                    ]
                }
            ],
            timeout=30
        )

        return {
            'success': True,
            'text': completion.choices[0].message.content
        }
    except Exception as e:
        return {
            'success': False,
            'text': f"识别失败: {str(e)}"
        }


def main():
    st.title("🤖 千问大模型 OCR 识别工具")
    st.markdown("基于 Qwen-VL-OCR 模型，高精度识别图片文字，支持复杂排版和手写体。")

    # 检查API Key是否已配置
    if API_KEY == "sk-" or not API_KEY:
        st.error("⚠️ 请先在环境变量中配置 DASHSCOPE_API_KEY")
        st.markdown("**获取API Key:** [阿里云百炼平台](https://www.aliyun.com/product/bailian)")
        return

    # 初始化客户端（使用与主项目一致的国内 DashScope 兼容模式地址）
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception as e:
        st.error(f"客户端初始化失败: {str(e)}")
        return

    # 侧边栏（高级选项）
    with st.sidebar:
        st.header("🔧 高级设置")

        custom_prompt = st.text_area(
            "自定义提示词（可选）",
            value="请提取这张图片中的所有文字内容，保持原始排版。只输出提取到的文字，不要添加任何解释。",
            height=150,
            help="可以自定义识别要求，例如：\"请提取发票中的金额、日期和发票号\"。"
        )

        st.markdown("---")

        # 使用说明
        st.header("📖 使用说明")
        st.markdown(
            """
        1. 上传一张或多张图片
        2. 点击“开始识别”按钮
        3. 在按钮下方查看“图片 + 文字”结果
        4. 可以导出全部识别结果为 CSV（Excel 可直接打开）

        **支持的格式：** PNG、JPG、JPEG、BMP、TIFF
        """
        )

        st.markdown("---")
        st.caption("提示：使用千问大模型会产生 API 调用费用，请注意控制调用量。")

        # 显示API状态
        st.success("✅ API 已配置")

    # 主区域：上传 + 识别按钮 + 结果
    st.subheader("📤 上传图片并识别")

    uploaded_files = st.file_uploader(
        "选择图片文件（可多选）",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        help="可以在文件选择窗口中多选图片（Windows 可进入文件夹后按 Ctrl+A 全选）。",
    )

    # 汇总所有待识别的图片：[(文件名, 二进制内容), ...]
    image_items: list[tuple[str, bytes]] = []

    if uploaded_files:
        for uf in uploaded_files:
            try:
                content = uf.read()
                if content:
                    image_items.append((uf.name, content))
            except Exception:
                continue

    if image_items:
        st.success(f"已检测到 {len(image_items)} 张图片。")

        # 识别按钮
        if st.button("🔍 开始识别", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 清空之前的结果
            st.session_state.qwen_results = {}

            total = len(image_items)
            # 批量处理
            for i, (name, img_bytes) in enumerate(image_items):
                status_text.text(f"正在识别: {name}")

                # 显示当前处理进度
                with st.spinner(f"正在处理第 {i + 1}/{total} 张图片..."):
                    # 处理图片
                    result = process_with_qwen(
                        client,
                        img_bytes,
                        custom_prompt
                        if custom_prompt
                        != "请提取这张图片中的所有文字内容，保持原始排版。只输出提取到的文字，不要添加任何解释。"
                        else None,
                    )

                    # 保存结果
                    try:
                        image_obj = Image.open(io.BytesIO(img_bytes))
                    except Exception:
                        image_obj = None

                    st.session_state.qwen_results[name] = {
                        "filename": name,
                        "text": result["text"],
                        "success": result["success"],
                        "image": image_obj,
                    }

                # 更新进度
                progress_bar.progress((i + 1) / total)
                time.sleep(0.05)  # 略微间隔，避免请求过快

            progress_bar.empty()
            status_text.empty()
            st.success("✅ 所有图片识别完成！")
    else:
        st.info("👆 请先上传至少一张图片文件。")

    # 在“开始识别”按钮下方直接展示结果
    if st.session_state.qwen_results:
        st.markdown("---")
        st.subheader("📋 识别结果")

        # 统计信息
        success_count = sum(1 for r in st.session_state.qwen_results.values() if r["success"])
        total_count = len(st.session_state.qwen_results)
        st.caption(f"成功: {success_count}/{total_count}")

        # 工具栏：导出 CSV（Excel 可直接打开）、清空
        tool_col1, tool_col2 = st.columns([1, 1])

        with tool_col1:
            # 构建 CSV 并导出（Excel 可直接打开）
            if st.button("💾 导出识别结果为 CSV（Excel 可打开）", use_container_width=True):
                rows: list[list[str]] = [["filename", "text"]]
                for filename, result in st.session_state.qwen_results.items():
                    if result["success"]:
                        rows.append([filename, result["text"]])

                if len(rows) > 1:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    for row in rows:
                        writer.writerow(row)
                    csv_bytes = output.getvalue().encode("utf-8-sig")
                    st.download_button(
                        "📥 点击下载 CSV 文件",
                        data=csv_bytes,
                        file_name="ocr_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.warning("当前没有成功的识别结果可导出")

        with tool_col2:
            if st.button("🗑️ 清空全部结果", use_container_width=True):
                st.session_state.qwen_results = {}
                st.experimental_rerun()

        st.markdown("---")

        # 逐条展示：一列图片，一列文字（就在按钮下面）
        for filename, result in st.session_state.qwen_results.items():
            st.markdown(f"#### 📷 {filename}")
            img_col, text_col = st.columns([1, 1])

            with img_col:
                st.image(result["image"], use_container_width=True)

            with text_col:
                if result["success"]:
                    st.text_area(
                        "识别结果",
                        value=result["text"],
                        height=220,
                        key=f"text_{filename}",
                    )
                else:
                    st.error(result["text"])
            st.markdown("---")


if __name__ == "__main__":
    main()