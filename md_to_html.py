#!/usr/bin/env python3
"""Convert README.md to HTML for PDF printing"""

import markdown

# Read markdown file
with open('README.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'toc']
)

# Create full HTML with styling
full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cats vs Dogs Classification - MLOps Pipeline</title>
    <style>
        @media print {{
            body {{ font-size: 11pt; }}
            pre {{ page-break-inside: avoid; }}
            h1, h2, h3 {{ page-break-after: avoid; }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{ color: #34495e; }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            font-size: 0.85em;
        }}
        pre code {{ background: none; padding: 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        a {{ color: #3498db; text-decoration: none; }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
        }}
        img {{ max-width: 100%; height: auto; }}
        ul, ol {{ padding-left: 25px; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

# Save HTML
with open('README.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("✅ Created README.html")
print("📄 To convert to PDF:")
print("   1. Open README.html in Safari or Chrome")
print("   2. Press Cmd+P (Print)")
print("   3. Select 'Save as PDF'")
