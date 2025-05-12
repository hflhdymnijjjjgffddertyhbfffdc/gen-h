import openai
import time
import logging
from datetime import datetime
import os
import PyPDF2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from webdriver_manager.core.driver_cache import DriverCacheManager
import subprocess
from audio import process_posters



# 环境变量配置
os.environ["WDM_PROXY"] = "https://registry.npmmirror.com/-/binary/chromedriver"
os.environ["WDM_LOCAL"] = "1"
os.environ["WDM_LOG"] = "false"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def process_streaming_response(completion):
    """处理流式响应"""
    msg = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.get("content"):
            content = chunk.choices[0].delta.get("content")
            msg += content
    return msg, completion

def ask_gpt(query, streaming_flg=True, max_retries=5, flag=True):
    """调用OpenAI API生成响应"""
    if flag:
        openai.api_base = "https://opus.gptuu.com/v1"
        openai.api_key = "sk-G0TnsMYO17kGKpQn0ScLk6xvtL72iKCiFkM4CSGfhxNRIQR6"
    else:
        openai.api_base = "https://openkey.cloud/v1"
        openai.api_key = "sk-JhthldK6DOHIqh7HA29e61451116419aA96d528b22886604"
    
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{'role': 'user', 'content': query}],
                stream=streaming_flg
            )
            msg = None

            if streaming_flg:
                msg, completion = process_streaming_response(completion)
            else:
                if not hasattr(completion, 'choices') or not completion.choices:
                    raise ValueError("No choices returned from API.")
                msg = completion.choices[0].message['content']

            if msg:
                end_time = time.time()
                logging.info(f"耗时: {end_time - start_time:.2f} 秒    GPT output: {msg}")
                time.sleep(3)
                return msg, completion

        except Exception as err:
            logging.warning(f'OpenAI API Error: {str(err)}')
            if attempt < max_retries:
                logging.warning(f"Retrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(2**(attempt + 2))
            else:
                if flag:
                    logging.error("Max retries reached. Switching to the other OpenAI API.")
                    return ask_gpt(query, streaming_flg, flag=False)
                else:
                    logging.error("All API attempts failed. Please check your network or API configuration.")
                    return None, None

def save_as_md(pdf_text, title, filename):
    """将PDF文本保存为Markdown文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n{pdf_text}")

def extract_title_from_md(md_path):
    """从Markdown文件提取标题（增强版）"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 优先提取Markdown标题
            match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'[\\/*?:"<>|]', '', title)
                return re.sub(r'\b(v\d+|final|draft)\b', '', title, flags=re.IGNORECASE)[:120]
            
            # 回退到原始提取逻辑
            lines = [ln.strip() for ln in content.split('\n')[:3] if len(ln.strip()) > 10]
            return lines[0][:120] if lines else None
    except Exception as e:
        logging.error(f"读取MD文件失败: {str(e)}")
        return None

def smart_wrap(title, max_line=2, chars_per_line=35):
    """智能多行标题处理器（匹配图片中的换行逻辑）"""
    import regex as re
    
    # 预处理
    title = re.sub(r'[\x00-\x1F\\/*?："<>|]', '', str(title))
    title = re.sub(r'\s+', ' ', title).strip()
    
    # 英文优先在介词后换行
    prep_break_points = [' for ', ' in ', ' with ', ' on ']
    for prep in prep_break_points:
        if prep in title.lower():
            index = title.lower().find(prep)
            if 10 < index < len(title)-10:  # 确保不在首尾断开
                return f"{title[:index+len(prep)]}<br>{title[index+len(prep):]}"
    
    # 中文换行逻辑
    if re.search(r'[\p{Han}]', title):
        return re.sub(r'(.{15})', '\\1<br>', title, count=1)  # 中文每15字换行
    
    # 英文单词换行
    words = title.split()
    if len(words) > 3:
        midpoint = len(words) // 2
        return ' '.join(words[:midpoint]) + '<br>' + ' '.join(words[midpoint:])
    
    return title

def generate_html_content(summary, article_title, author_name):
    """生成科创板日报风格的HTML海报"""
    current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S 星期") + ["一", "二", "三", "四", "五", "六", "日"][datetime.now().weekday()]
    
    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Paper Digest</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');
        body {{
            font-family: 'Noto Sans SC', sans-serif;
            background-color: #e6f0fa;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            overflow: hidden;
        }}
        .poster {{
            width: 1242px;
            height: 1660px;
            background: #ffffff;
            border-radius: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            position: relative;
            display: flex;
            flex-direction: column;
            
        }}
        .header {{
            padding: 40px;
            background: #e6f0fa;
            text-align: left;
        }}
        .header img {{
            width: 80px;
            height: 80px;
            vertical-align: middle;
            margin-right: 20px;
        }}
        .header h1 {{
            display: inline;
            font-size: 48px;
            font-weight: 700;
            color: #1a3c6e;
        }}
        .header p {{
            font-size: 32px;
            color: #1a3c6e;
            margin-top: 10px;
        }}
        .content {{
            flex: 1;
            padding: 40px;
            background: #ffffff;
            color: #333;
        }}
        .content .date {{
            font-size: 28px;
            color: #666;
            margin-bottom: 20px;
        }}
        .content h2 {{
            font-size: 56px;
            font-weight: 700;
            color: #1a3c6e;
            margin-bottom: 20px;
        }}
        .content p {{
            font-size: 32px;
            line-height: 1.6;
            color: #333;
            margin-bottom: 30px;
        }}
        .buttons {{
            display: flex;
            gap: 20px;
            margin-bottom: 40px;
        }}
        .buttons a {{
            padding: 10px 20px;
            background: #e6f0fa;
            color: #1a3c6e;
            font-size: 24px;
            border-radius: 8px;
            text-decoration: none;
            transition: background-color 0.3s;
        }}
        .buttons a:hover {{
            background: #d1e3f6;
        }}
        .footer {{
            padding: 40px;
            background: #ffffff;
            border-top: 1px solid #e6f0fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .footer .left {{
            display: flex;
            align-items: center;
        }}
        .footer .left a {{
            font-size: 32px;
            color: #1a3c6e;
            text-decoration: none;
            margin-right: 20px;
        }}
        .footer .left span {{
            font-size: 24px;
            color: #666;
        }}
        .footer .right img {{
            width: 120px;
            height: 120px;
        }}
        .bottom-bar {{
            padding: 20px;
            background: #1a3c6e;
            color: #ffffff;
            font-size: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="poster">
        <div class="header">
            
            <h1>学术海报</h1>
            <p></p>
        </div>
        <div class="content">
            <div class="date">{current_date}</div>
            <h2>{article_title}</h2>
            <p>{summary}</p>
            <div class="buttons">
                <a href="#">原创行业观察</a>
                <a href="#">学术推送</a>
                <a href="#">深度</a>
            </div>
        </div>
        <div class="footer">
            <div class="left">
                <a href="#">星电报</a>
                
            </div>
            <div class="right">
               
            </div>
        </div>
        <div class="bottom-bar">
            上海理工大学出品
        </div>
    </div>
</body>
</html>
"""

# def html_to_image(html_filename, image_filename):
#     """将HTML文件转为科创板日报风格的长截图"""
#     def get_chrome_version():
#         try:
#             output = subprocess.check_output(
#                 ["google-chrome", "--version"],
#                 stderr=subprocess.STDOUT,
#                 timeout=5
#             ).decode('utf-8', errors='ignore')
#             return re.search(r'\d+\.\d+\.\d+', output).group()
#         except Exception as e:
#             logging.warning(f"浏览器检测失败: {str(e)}")
#             return None

#     driver = None
#     try:
#         service = Service(
#             ChromeDriverManager().install(),
#             log_output=os.devnull
#         )
#         options = webdriver.ChromeOptions()
#         options.add_argument('--headless=new')
#         options.add_argument('--disable-gpu')
#         options.add_argument('--window-size=1242,1800')
#         driver = webdriver.Chrome(service=service, options=options)

#         # 加载页面
#         abs_path = os.path.abspath(html_filename)
#         driver.get(f"file://{abs_path}?t={int(time.time())}")

#         # 关键修复点：补全WebDriverWait的括号
#         WebDriverWait(driver, 15).until(
#             EC.presence_of_element_located((By.CLASS_NAME, 'bottom-bar'))
#         )  # ← 此处必须闭合括号

#         # 增强版高度计算
#         total_height = driver.execute_script("""\
#             try {
#                 const bodyScroll = document.body.scrollHeight || 0;
#                 const docScroll = document.documentElement.scrollHeight || 0;
#                 const docOffset = document.documentElement.offsetHeight || 0;
#                 return Math.max(bodyScroll, docScroll, docOffset) + 50;
#             } catch(e) {
#                 return document.documentElement.scrollHeight;
#             }\
#         """)

#         driver.set_window_size(1242, min(total_height, 10000))
#         driver.execute_script("window.scrollTo(0, 0)")
#         time.sleep(0.3)
#         driver.save_screenshot(image_filename)
#         return True

#     except Exception as e:
#         logging.error(f"截图流程异常: {str(e)}")
#         return False
#     finally:
#         if driver:
#             driver.quit()

def html_to_image(html_filename, image_filename):
    """将HTML文件转为科创板日报风格的长截图"""
    driver = None
    try:
        # 手动指定ChromeDriver路径（替换为您的实际路径）
        chrome_driver_path = r"D:\代码\.wdm\drivers\chromedriver\win64\136.0.7103.92\chromedriver.exe"
        
        service = Service(executable_path=chrome_driver_path)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1242,1800')
        driver = webdriver.Chrome(service=service, options=options)

        # 加载页面
        abs_path = os.path.abspath(html_filename)
        driver.get(f"file://{abs_path}?t={int(time.time())}")

        # 等待页面加载完成
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'bottom-bar'))
        )

        # 计算页面高度
        total_height = driver.execute_script("""
            try {
                const bodyScroll = document.body.scrollHeight || 0;
                const docScroll = document.documentElement.scrollHeight || 0;
                const docOffset = document.documentElement.offsetHeight || 0;
                return Math.max(bodyScroll, docScroll, docOffset) + 50;
            } catch(e) {
                return document.documentElement.scrollHeight;
            }
        """)

        # 设置窗口大小并截图
        driver.set_window_size(1242, min(total_height, 10000))
        driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(0.3)
        driver.save_screenshot(image_filename)
        return True

    except Exception as e:
        logging.error(f"截图流程异常: {str(e)}")
        return False
    finally:
        if driver:
            driver.quit()



def main():
    """主处理流程"""



    pdf_folder = r"D:\代码\pdf"
    md_folder = r"D:\代码\md"
    html_folder = r"D:\代码\html"
    image_folder = r"D:\代码\posters"

    # pdf_folder = "/home/asus/HuFan/ppt/pdf"
    # md_folder = "/home/asus/HuFan/ppt/md"
    # html_folder = "/home/asus/HuFan/ppt/html"
    # image_folder = "/home/asus/HuFan/ppt/posters"
    #output_video = r"D:\代码\audio\output.mp4"
    
    os.makedirs(md_folder, exist_ok=True)
    os.makedirs(html_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            # PDF转MD
            md_path = os.path.join(md_folder, f"{base_name}.md")
            pdf_text = ""
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pdf_text = '\n'.join([page.extract_text() for page in reader.pages])
                
                # 保存MD文件
                initial_title = filename.rsplit('.', 1)[0].replace('_', ' ')
                save_as_md(pdf_text, initial_title, md_path)
                logging.info(f"已生成MD文件: {md_path}")
            except Exception as e:
                logging.error(f"PDF转换失败: {str(e)}")
                continue
            
            # 从MD提取标题
            article_title = extract_title_from_md(md_path) or initial_title
            final_title = smart_wrap(article_title)
            logging.info(f"最终标题: {final_title}")
            
            # 生成摘要和海报
            html_filename = os.path.join(html_folder, f"{base_name}_summary.html")
            image_filename = os.path.join(image_folder, f"{base_name}_summary.png")
            
            try:
                prompt = f"""将论文生成一段话，需要包含背景、技术创新、结果、句子之间连贯，专业术语少一些，实验细节尽可能的详细，生成的整个段落尽可能的简洁，要求符合新闻播报的风格，直接输出内容。

论文内容：
{pdf_text}
                    """
                
                result, _ = ask_gpt(prompt, streaming_flg=False)
                if result:
                    html_content = generate_html_content(result, final_title, "AI Research")
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    html_to_image(html_filename, image_filename)
                else:
                    logging.error("摘要生成失败")
            except Exception as e:
                logging.error(f"处理异常: {str(e)}")
            
            logging.info(f"处理完成: {filename}\n{'='*60}")

    # 调用之前生成的接口
   
    output_ppt_path = r"D:\代码\pptx\海报.pptx"
    output_video_path = r"D:\代码\audio\final_output.mp4"
    

    process_posters(html_folder,image_folder, output_ppt=output_ppt_path, output_video=output_video_path)


if __name__ == "__main__":
    main()