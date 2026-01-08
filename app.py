#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Face Fixer - الإصدار النهائي المستقر
نسخة مبسطة ومضمونة العمل على HuggingFace Spaces
"""

import sys
import os
import time
import logging
import warnings
warnings.filterwarnings('ignore')

# 1. إعداد المسارات والبيئة
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'

# إنشاء المجلدات المطلوبة
os.makedirs('/tmp/torch_cache', exist_ok=True)
os.makedirs('/tmp/huggingface_cache', exist_ok=True)

# 2. إصلاحات التوافق
import torchvision
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

# 3. استيراد المكتبات الأساسية
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageFilter
import torch

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 4. الخوارزمية الأساسية (محفوظة كما هي)
def smart_restore_perfectionist(input_img, strength=1.0):
    """
    الخوارزمية الرئيسية لترميم الوجه - محفوظة تماماً كما هي
    """
    if input_img is None: 
        return None, "⚠️ الرجاء تحميل صورة أولاً"
    
    try:
        start_time = time.time()
        
        # تحويل الصورة من numpy (Gradio) إلى BGR (OpenCV)
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        
        # تسجيل الحجم الأصلي
        original_h, original_w = img.shape[:2]
        
        # تقليل الحجم إذا كان كبيراً
        if original_w > 1000 or original_h > 1000:
            scale = min(1000 / original_w, 1000 / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # تحميل النموذج
        try:
            from gfpgan import GFPGANer
            
            # استخدام النموذج المدمج (سيتحمله GFPGAN تلقائياً)
            face_enhancer = GFPGANer(
                model_path='GFPGANv1.4',
                upscale=1.5,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
        except Exception as e:
            logger.error(f"Error loading GFPGAN: {e}")
            return None, f"❌ خطأ في تحميل النموذج: {str(e)}"
        
        # خوارزمية Ultimate Balance الأصلية (غير ملموسة)
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        silk = cv2.edgePreservingFilter(output, flags=1, sigma_s=30, sigma_r=0.08)
        lab = cv2.cvtColor(silk, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.addWeighted(l, 1.1, cv2.GaussianBlur(l, (0,0), 3), -0.1, 0)
        final_ai = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        img_upscaled = cv2.resize(img, (output.shape[1], output.shape[0]))
        inter_mix = cv2.addWeighted(img_upscaled, 0.5, silk, 0.5, 0)
        
        # تطبيق قوة التحسين
        alpha = 0.8 * strength
        beta = 0.2 * strength
        final = cv2.addWeighted(inter_mix, alpha, final_ai, beta, 0)
        
        # التحويل النهائي
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        # إحصائيات المعالجة
        processing_time = time.time() - start_time
        stats = f"""
✅ تمت المعالجة بنجاح!

📊 إحصائيات المعالجة:
• الحجم الأصلي: {original_w}×{original_h}
• الحجم الناتج: {final.shape[1]}×{final.shape[0]}
• قوة التحسين: {strength}
• وقت المعالجة: {processing_time:.2f} ثانية
• الجهاز: {'GPU' if torch.cuda.is_available() else 'CPU'}

💡 ملاحظة: تم تطبيق خوارزمية Ultimate Balance الأصلية
        """
        
        return final_rgb, stats
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None, f"❌ خطأ في المعالجة: {str(e)}"

# 5. إنشاء الواجهة
def create_interface():
    """إنشاء واجهة متوافقة"""
    
    # CSS مبسط
    custom_css = """
    :root {
        --primary: #1c4167;
        --secondary: #007eff;
        --accent: #ff6b6b;
    }
    
    .gradio-container {
        max-width: 1000px;
        margin: auto;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    .header {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        padding: 30px;
        text-align: center;
        color: white;
    }
    
    .header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 800;
    }
    
    .header p {
        margin: 10px 0 0;
        opacity: 0.9;
        font-size: 1.1em;
    }
    
    .content {
        padding: 30px;
    }
    
    .image-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    @media (max-width: 768px) {
        .image-row {
            grid-template-columns: 1fr;
        }
    }
    
    .image-box {
        border: 3px dashed #ddd;
        border-radius: 15px;
        padding: 15px;
        background: #f8f9fa;
    }
    
    .controls {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #e2e8f0;
    }
    
    .process-btn {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border: none;
        color: white;
        padding: 15px 30px;
        font-size: 1.2em;
        font-weight: bold;
        border-radius: 10px;
        cursor: pointer;
        width: 100%;
        margin-top: 10px;
        transition: all 0.3s;
    }
    
    .process-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(28, 65, 103, 0.2);
    }
    
    .stats-box {
        background: #e8f4ff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        font-family: monospace;
        white-space: pre-wrap;
        border-left: 5px solid var(--secondary);
    }
    
    footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.9em;
        border-top: 1px solid #eee;
        margin-top: 30px;
    }
    """
    
    # بناء الواجهة
    with gr.Blocks(css=custom_css, title="Ultimate Face Fixer") as demo:
        
        # الرأس
        gr.HTML("""
            <div class="header">
                <h1>✨ Ultimate Face Fixer</h1>
                <p>ترميم وتجميل الصور بتقنية الذكاء الاصطناعي المتطورة</p>
                <div style="margin-top: 10px; font-size: 0.9em;">
                    <span>الإصدار المستقر | متوافق كلياً مع HuggingFace</span>
                </div>
            </div>
        """)
        
        # المحتوى الرئيسي
        with gr.Column(elem_classes="content"):
            
            # قسم الصور
            with gr.Row(elem_classes="image-row"):
                # الصورة المدخلة
                with gr.Column(elem_classes="image-box"):
                    gr.Markdown("### 📤 الصورة الأصلية")
                    input_image = gr.Image(type="numpy", height=320)
                
                # الصورة الناتجة
                with gr.Column(elem_classes="image-box"):
                    gr.Markdown("### 📥 الصورة المحسنة")
                    output_image = gr.Image(type="numpy", height=320)
            
            # عناصر التحكم
            with gr.Column(elem_classes="controls"):
                gr.Markdown("### ⚙️ إعدادات المعالجة")
                
                # شريط قوة التحسين
                strength_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="قوة التحسين",
                    info="من خفيف (0.5) إلى قوي (2.0)"
                )
                
                # زر المعالجة
                process_btn = gr.Button(
                    "🚀 بدء الترميم",
                    variant="primary",
                    size="lg",
                    elem_classes="process-btn"
                )
                
                # رسالة الحالة
                status_msg = gr.Textbox(
                    label="حالة المعالجة",
                    value="⚡ جاهز للبدء - قم بتحميل صورة",
                    interactive=False
                )
            
            # الإحصائيات
            stats_output = gr.Textbox(
                label="📊 نتائج المعالجة",
                lines=8,
                interactive=False,
                elem_classes="stats-box"
            )
            
            # التعليمات
            with gr.Accordion("📖 دليل الاستخدام السريع", open=False):
                gr.Markdown("""
                ### خطوات الاستخدام:
                1. **قم بتحميل صورة** عن طريق السحب والإفلات أو النقر على منطقة الرفع
                2. **اضبط قوة التحسين** باستخدام شريط التمرير (1.0 هو المستوى الأمثل)
                3. **انقر على زر "بدء الترميم"**
                4. **انتظر** حتى تظهر النتيجة (عادة 10-30 ثانية)
                5. **تحقق من الإحصائيات** في الأسفل
                
                ### ⚠️ ملاحظات هامة:
                - الخوارزمية الأساسية محفوظة تماماً كما هي
                - يدعم الصيغ: JPG, PNG, JPEG, BMP
                - الحد الأقصى لحجم الصورة: 2000×2000 بكسل
                - الصور الكبيرة جداً يتم تصغيرها تلقائياً
                - المعالجة الأولى قد تستغرق وقتاً أطول لتحميل النموذج
                """)
            
            # التذييل
            gr.HTML("""
                <footer>
                    <p>Ultimate Face Fixer | تم التطوير باستخدام GFPGAN</p>
                    <p style="font-size: 0.8em; color: #888;">
                        ملاحظة: الخوارزمية الأساسية لتحسين الوجه محفوظة تماماً كما هي
                    </p>
                </footer>
            """)
        
        # ربط الأحداث
        def process_image(image, strength):
            if image is None:
                return None, "⚠️ الرجاء تحميل صورة أولاً", ""
            
            result, stats = smart_restore_perfectionist(image, strength)
            
            if result is None:
                return None, "❌ فشل في معالجة الصورة", stats
            else:
                return result, "✅ تمت المعالجة بنجاح!", stats
        
        process_btn.click(
            fn=process_image,
            inputs=[input_image, strength_slider],
            outputs=[output_image, status_msg, stats_output]
        )
        
        # تلميحات تفاعلية
        input_image.change(
            fn=lambda x: "📸 الصورة جاهزة للمعالجة!" if x is not None else "⚡ جاهز للبدء - قم بتحميل صورة",
            inputs=[input_image],
            outputs=[status_msg]
        )
    
    return demo

# 6. الدالة الرئيسية
def main():
    """الدالة الرئيسية للتشغيل"""
    print("=" * 60)
    print("Ultimate Face Fixer - الإصدار المستقر")
    print("=" * 60)
    
    # إنشاء الواجهة
    print("🚀 جاري تحميل الواجهة...")
    demo = create_interface()
    
    # تشغيل الواجهة
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

# 7. نقطة الدخول الرئيسية
if __name__ == "__main__":
    main()