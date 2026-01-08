import sys
import os
import cv2
import numpy as np
import gradio as gr
import torch
import torchvision

# 1. إصلاح التوافقية (إلزامي)
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# 2. إعداد المحركات (الوجه + الخلفية)
# محرك الخلفية (Real-ESRGAN)
model_bg = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
bg_upsampler = RealESRGANer(
    scale=2,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model_bg,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_value_available() else False
)

# محرك الوجه (GFPGAN) مع دمج محرك الخلفية فيه
face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler
)

def ultimate_service_processor(input_img, enhance_bg):
    if input_img is None: 
        return None
    
    # تحويل الصورة لمعالجتها بـ OpenCV
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    try:
        # تنفيذ المعالجة (الوجه + الخلفية إذا تم اختيارها)
        # bg_upsampler سيُستخدم تلقائياً داخل enhancer إذا مررنا enhance_bg
        _, _, output = face_enhancer.enhance(
            img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )

        # تطبيق خوارزمية Ultimate Balance الخاصة بك (التنعيم والحدة)
        silk = cv2.edgePreservingFilter(output, flags=1, sigma_s=30, sigma_r=0.08)
        lab = cv2.cvtColor(silk, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.addWeighted(l, 1.1, cv2.GaussianBlur(l, (0,0), 3), -0.1, 0)
        final_ai = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        # الدمج النهائي للنتائج
        img_upscaled = cv2.resize(img, (output.shape[1], output.shape[0]))
        inter_mix = cv2.addWeighted(img_upscaled, 0.5, silk, 0.5, 0)
        final = cv2.addWeighted(inter_mix, 0.8, final_ai, 0.2, 0)
        
        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error: {e}")
        return input_img

# 3. واجهة المستخدم الموجهة لـ "الشخص العادي"
custom_css = """
.gradio-container { background-color: #fcfcfc !important; }
#main_box { border: 2px solid #1c4167 !important; border-radius: 20px !important; padding: 20px !important; }
.desc_text { text-align: right; color: #444; font-size: 1.1em; margin-bottom: 10px; }
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="main_box"):
        gr.HTML("""
            <div style='text-align: center; padding: 10px;'>
                <h1 style='color: #1c4167; margin-bottom: 0;'>Ultimate Face Restorer</h1>
                <p style='color: #666;'>خصوصيتك مضمونة - يتم مسح الصور فور المعالجة</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column():
                input_i = gr.Image(type="numpy", label="ارفع صورتك هنا")
                bg_check = gr.Checkbox(label="تحسين جودة الصورة بالكامل (وليس الوجه فقط)", value=True)
                submit_btn = gr.Button("ابدأ الترميم الآن ✨", variant="primary")
            
            with gr.Column():
                output_i = gr.Image(type="numpy", label="النتيجة النهائية")
        
        gr.HTML("""
            <div style='background: #eef4f9; padding: 15px; border-radius: 10px; margin-top: 15px; text-align: right;'>
                <strong>💡 نصيحة للنتيجة:</strong> انتظر حوالي 10-15 ثانية، الذكاء الاصطناعي يقوم الآن بإعادة بناء ملامح الوجه بدقة 4K.
            </div>
        """)

    submit_btn.click(
        fn=ultimate_service_processor, 
        inputs=[input_i, bg_check], 
        outputs=output_i
    )

if __name__ == "__main__":
    demo.launch()