# Academic Poster Generator

An automated pipeline for converting academic PDFs into visually appealing posters and videos.

## 🚀 Features

- **PDF Processing**: Extracts text from academic PDFs
- **AI Summarization**: Uses GPT-4 to generate concise summaries of research papers
- **Poster Generation**: Creates professional HTML posters with responsive design
- **Video Creation**: Converts posters into video format with audio narration
- **Multi-platform Support**: Works on both Windows and Linux systems

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google Chrome (for screenshot functionality)
- FFmpeg (for video processing)

### Setup
```bash
pip install -r requirements.txt
🛠 Configuration
Set up your OpenAI API keys in the script or as environment variables
Configure the folder paths in main() function:
pdf_folder = "path/to/your/pdf/folder"
md_folder = "path/to/markdown/output"
html_folder = "path/to/html/output"
image_folder = "path/to/image/output"
🏃 Usage
Run the main script:
python gen_htm.py
The pipeline will:

Process all PDFs in the input folder

Generate markdown files

Create HTML posters

Convert posters to images

Generate a final PowerPoint and video
📂 Project Structure
.
├── audio2.py            # Audio and video processing module
├── main.py              # Main processing pipeline
├── requirements.txt     # Python dependencies
├── pdf/                 # Input PDF folder
├── md/                  # Markdown output
├── html/                # HTML posters
├── posters/             # Generated poster images
├── pptx/                # PowerPoint output
└── audio/               # Final video output
⚙️ Technical Details
Uses PyPDF2 for PDF text extraction

Implements GPT-4 for intelligent summarization

Selenium for HTML-to-image conversion

Responsive HTML/CSS for poster design

FFmpeg for video processing


