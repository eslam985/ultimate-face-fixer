import os
import cv2
import numpy as np
import gradio as gr
from gfpgan import GFPGANer

# 1. إعدادات المحرك
model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
face_enhancer = GFPGANer(model_path=model_url, upscale=1.5, arch='clean', channel_multiplier=2, bg_upsampler=None)

# 2. حقنة الـ CSS المتوافقة مع ألوان موقعك
custom_css = """
body { background-color: #f9f9f9 !important; }
.gradio-container { max-width: 850px !important; margin: auto !important; border: 1px solid #b3ccff !important; border-radius: 15px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; }
#title_area { text-align: center; color: #1c4167; padding: 25px; border-bottom: 2px solid #007eff30; }
.icon-button { transform: scale(1.8) !important; margin: 10px !important; color: #1c4167 !important; }
button.primary { 
    background: #1c4167 !important; 
    border: none !important; 
    color: #f9f9f9 !important; 
    font-weight: bold !important; 
    border-radius: 8px !important; 
    height: 55px !important;
    transition: background 0.3s ease !important;
}
button.primary:hover { background: #005bb5 !important; }
footer {display: none !important;}
.image-container { border: 2px solid #b3ccff !important; border-radius: 10px !important; }
"""

def smart_restore_perfectionist(input_img_path):
    if input_img_path is None: 
        return None
    
    # قراءة الصورة من المسار لتقليل ضغط الرام
    img = cv2.imread(input_img_path)
    if img is None: 
        return None
        
    h, w = img.shape[:2]
    if w > 2000 or h > 2000:
        img = cv2.resize(img, (w // 2, h // 2))
        
    try:
        # خوارزمية Ultimate Balance
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        silk = cv2.edgePreservingFilter(output, flags=1, sigma_s=30, sigma_r=0.08)
        lab = cv2.cvtColor(silk, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.addWeighted(l, 1.1, cv2.GaussianBlur(l, (0,0), 3), -0.1, 0)
        final_ai = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        img_upscaled = cv2.resize(img, (output.shape[1], output.shape[0]))
        inter_mix = cv2.addWeighted(img_upscaled, 0.5, silk, 0.5, 0)
        final = cv2.addWeighted(inter_mix, 0.8, final_ai, 0.2, 0)
        
        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error: {e}")
        return None

# 3. بناء الواجهة
with gr.Blocks() as demo:
    with gr.Column(elem_id="title_area"):
        gr.HTML("""
            <h1 style='color: #1c4167; font-size: 2.2em; margin-bottom: 5px;'>Ultimate Face Restorer</h1>
            <p style='color: #666; font-size: 1.1em;'>ترميم ملامح الوجه بتقنية Ultimate Balance</p>
        """)
    
    gr.Interface(
        fn=smart_restore_perfectionist,
        inputs=gr.Image(type="filepath", label="ارفع الصورة القديمة"),
        outputs=gr.Image(type="numpy", label="النتيجة النهائية")
    )

if __name__ == "__main__":
    # التعديل الحاسم: حذفنا show_api وحافظنا على ssr_mode=False
    demo.launch(css=custom_css, ssr_mode=False)