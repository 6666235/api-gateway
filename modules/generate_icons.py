"""
生成 PWA 图标
"""
import os
from PIL import Image, ImageDraw, ImageFont

def generate_icon(size: int, output_path: str):
    """生成指定尺寸的图标"""
    # 创建图像
    img = Image.new('RGBA', (size, size), (99, 102, 241, 255))  # 主题色
    draw = ImageDraw.Draw(img)
    
    # 绘制圆角矩形背景
    radius = size // 8
    draw.rounded_rectangle(
        [(0, 0), (size - 1, size - 1)],
        radius=radius,
        fill=(99, 102, 241, 255)
    )
    
    # 绘制 AI 文字
    font_size = size // 3
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = "AI"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - size // 10
    
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    # 绘制 Hub 文字
    hub_font_size = size // 6
    try:
        hub_font = ImageFont.truetype("arial.ttf", hub_font_size)
    except:
        hub_font = ImageFont.load_default()
    
    hub_text = "Hub"
    hub_bbox = draw.textbbox((0, 0), hub_text, font=hub_font)
    hub_width = hub_bbox[2] - hub_bbox[0]
    
    hub_x = (size - hub_width) // 2
    hub_y = y + text_height + size // 20
    
    draw.text((hub_x, hub_y), hub_text, fill=(255, 255, 255, 200), font=hub_font)
    
    # 保存
    img.save(output_path, 'PNG')
    print(f"Generated: {output_path}")


def generate_simple_icon(size: int, output_path: str):
    """生成简单的图标（不依赖字体）"""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 圆角矩形背景
    radius = size // 6
    draw.rounded_rectangle(
        [(0, 0), (size - 1, size - 1)],
        radius=radius,
        fill=(99, 102, 241, 255)
    )
    
    # 绘制机器人图案
    center = size // 2
    
    # 头部（圆形）
    head_radius = size // 4
    draw.ellipse(
        [(center - head_radius, size // 4 - head_radius // 2),
         (center + head_radius, size // 4 + head_radius + head_radius // 2)],
        fill=(255, 255, 255, 255)
    )
    
    # 眼睛
    eye_radius = size // 16
    eye_y = size // 4 + head_radius // 4
    draw.ellipse(
        [(center - head_radius // 2 - eye_radius, eye_y - eye_radius),
         (center - head_radius // 2 + eye_radius, eye_y + eye_radius)],
        fill=(99, 102, 241, 255)
    )
    draw.ellipse(
        [(center + head_radius // 2 - eye_radius, eye_y - eye_radius),
         (center + head_radius // 2 + eye_radius, eye_y + eye_radius)],
        fill=(99, 102, 241, 255)
    )
    
    # 身体
    body_top = size // 4 + head_radius + size // 16
    body_width = size // 3
    body_height = size // 3
    draw.rounded_rectangle(
        [(center - body_width // 2, body_top),
         (center + body_width // 2, body_top + body_height)],
        radius=size // 16,
        fill=(255, 255, 255, 255)
    )
    
    # 天线
    antenna_height = size // 8
    draw.line(
        [(center, size // 4 - head_radius // 2),
         (center, size // 4 - head_radius // 2 - antenna_height)],
        fill=(255, 255, 255, 255),
        width=max(2, size // 32)
    )
    draw.ellipse(
        [(center - size // 24, size // 4 - head_radius // 2 - antenna_height - size // 24),
         (center + size // 24, size // 4 - head_radius // 2 - antenna_height + size // 24)],
        fill=(255, 255, 255, 255)
    )
    
    img.save(output_path, 'PNG')
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    
    # 生成不同尺寸的图标
    sizes = [192, 512]
    
    for size in sizes:
        output_path = os.path.join(static_dir, f'icon-{size}.png')
        generate_simple_icon(size, output_path)
    
    # 生成 favicon
    favicon_path = os.path.join(static_dir, 'favicon.ico')
    img = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0, 0), (31, 31)], radius=4, fill=(99, 102, 241, 255))
    # 简单的 AI 标记
    draw.rectangle([(8, 10), (12, 22)], fill=(255, 255, 255, 255))
    draw.rectangle([(20, 10), (24, 22)], fill=(255, 255, 255, 255))
    draw.rectangle([(12, 14), (20, 18)], fill=(255, 255, 255, 255))
    img.save(favicon_path, 'ICO')
    print(f"Generated: {favicon_path}")
    
    print("All icons generated!")
