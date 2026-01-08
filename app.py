import sys
# 1. حقنة الإصلاح الإجبارية (يجب أن تكون في أول سطر)
import torchvision
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

import os
import cv2
import numpy as np
import gradio as gr
from gfpgan import GFPGANer

# 2. إعدادات المحرك
model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
face_enhancer = GFPGANer(model_path=model_url, upscale=1.5, arch='clean', channel_multiplier=2, bg_upsampler=None)

# 3. الـ CSS الخاص بك
custom_css = """
body { background-color: #f9f9f9 !important; }
.gradio-container { max-width: 850px !important; margin: auto !important; border: 1px solid #b3ccff !important; border-radius: 15px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; }
#title_area { text-align: center; color: #1c4167; padding: 25px; border-bottom: 2px solid #007eff30; }
button.primary { 
    background: #1c4167 !important; 
    border: none !important; 
    color: #f9f9f9 !important; 
    font-weight: bold !important; 
    border-radius: 8px !important; 
    height: 55px !important;
}
footer {display: none !important;}
"""

def smart_restore_perfectionist(input_img):
    if input_img is None: 
        return None
    
    # تحويل الصورة من numpy (Gradio) إلى BGR (OpenCV)
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        
    h, w = img.shape[:2]
    if w > 2000 or h > 2000:
        img = cv2.resize(img, (w // 2, h // 2))
        
    try:
        # خوارزمية Ultimate Balance الخاصة بك
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

# 4. بناء الواجهة
with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_id="title_area"):
        gr.HTML("""
            <h1 style='color: #1c4167; font-size: 2.2em; margin-bottom: 5px;'>Ultimate Face Restorer</h1>
            <p style='color: #666; font-size: 1.1em;'>ترميم ملامح الوجه بتقنية Ultimate Balance</p>
        """)
    
    with gr.Row():
        input_i = gr.Image(type="numpy", label="ارفع الصورة القديمة")
        output_i = gr.Image(type="numpy", label="النتيجة النهائية")
    
    submit_btn = gr.Button("ابدأ الترميم الآن ✨", variant="primary")
    submit_btn.click(fn=smart_restore_perfectionist, inputs=input_i, outputs=output_i)

if __name__ == "__main__":
    demo.launch()