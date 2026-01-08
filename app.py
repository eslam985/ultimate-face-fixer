#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Face Fixer - الإصدار المعدل لـ Gradio 6.2.0 على HuggingFace
"""

import sys
import os
import time
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 1. إصلاحات التوافق
import torchvision
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

# 2. استيراد المكتبات الأساسية
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

# 3. إعداد المسارات
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'

# 4. مدير النموذج
class FaceRestorer:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """تحميل النموذج"""
        if self.model is not None:
            return self.model
        
        try:
            from gfpgan import GFPGANer
            
            # استخدام النموذج المدمج في HuggingFace
            self.model = GFPGANer(
                model_path='GFPGANv1.4',
                upscale=1.5,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            
            logger.info("✅ Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

# 5. الخوارزمية الأساسية (محفوظة كما هي)
def process_face_restoration(input_image, strength=1.0):
    """
    الخوارزمية الرئيسية لترميم الوجه - محفوظة تماماً كما هي
    """
    try:
        if input_image is None:
            return None, "⚠️ الرجاء تحميل صورة أولاً"
        
        start_time = time.time()
        
        # الحصول على مصفوفة الصورة (تتوافق مع Gradio 6.x)
        if isinstance(input_image, dict):
            img_array = input_image['image']
        else:
            img_array = input_image
        
        # تحويل الصورة
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        original_h, original_w = img.shape[:2]
        
        # تقليل الحجم إذا كان كبيراً
        if original_w > 1000 or original_h > 1000:
            scale = min(1000 / original_w, 1000 / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # تحميل النموذج
        restorer = FaceRestorer()
        model = restorer.load_model()
        
        # خوارزمية Ultimate Balance الأصلية (محفوظة كما هي)
        try:
            _, _, output = model.enhance(
                img, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True
            )
        except Exception as e:
            logger.warning(f"First enhance attempt failed: {e}, trying again...")
            _, _, output = model.enhance(
                img, 
                has_aligned=False, 
                only_center_face=True, 
                paste_back=True
            )
        
        # المعالجة التالية (خوارزمية محفوظة)
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
        
        # تحسين النهائي
        final_pil = Image.fromarray(final_rgb)
        if strength > 1.0:
            final_pil = final_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=100))
        
        final_array = np.array(final_pil)
        
        # إحصائيات
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
        
        return final_array, stats
        
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        return None, f"❌ خطأ في المعالجة: {str(e)}"

# 6. إنشاء الواجهة
def create_interface():
    """إنشاء واجهة متوافقة مع Gradio 6.2.0"""
    
    # CSS مبسط
    custom_css = """
    :root {
        --primary: #1c4167;
        --secondary: #007eff;
        --accent: #ff6b6b;
        --bg: #f9f9f9;
    }
    
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
        margin: 0 !important;
        padding: 20px !important;
        min-height: 100vh !important;
    }
    
    .gradio-container {
        max-width: 1000px !important;
        margin: 0 auto !important;
        background: white !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
        overflow: hidden !important;
        padding: 0 !important;
    }
    
    .header {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        padding: 30px !important;
        text-align: center !important;
        color: white !important;
        margin: 0 !important;
    }
    
    .header h1 {
        margin: 0 !important;
        font-size: 2.5em !important;
        font-weight: 800 !important;
    }
    
    .header p {
        margin: 10px 0 0 !important;
        opacity: 0.9 !important;
        font-size: 1.1em !important;
    }
    
    .content {
        padding: 30px !important;
    }
    
    .image-row {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 20px !important;
        margin-bottom: 30px !important;
    }
    
    @media (max-width: 768px) {
        .image-row {
            grid-template-columns: 1fr !important;
        }
    }
    
    .image-box {
        border: 3px dashed #ddd !important;
        border-radius: 15px !important;
        padding: 15px !important;
        background: #f8f9fa !important;
        min-height: 350px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    .controls {
        background: #f8f9fa !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin-bottom: 25px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .control-group {
        margin-bottom: 20px !important;
    }
    
    .process-btn {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        border: none !important;
        color: white !important;
        padding: 15px 30px !important;
        font-size: 1.2em !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        width: 100% !important;
        margin-top: 10px !important;
        transition: all 0.3s !important;
    }
    
    .process-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(28, 65, 103, 0.2) !important;
    }
    
    .stats-box {
        background: #e8f4ff !important;
        border-radius: 15px !important;
        padding: 20px !important;
        margin-top: 20px !important;
        font-family: monospace !important;
        white-space: pre-wrap !important;
        border-left: 5px solid var(--secondary) !important;
    }
    
    .features {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
        gap: 15px !important;
        margin-top: 30px !important;
    }
    
    .feature {
        background: #f0f7ff !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border-left: 4px solid var(--primary) !important;
    }
    
    .feature h4 {
        margin: 0 0 10px 0 !important;
        color: var(--primary) !important;
    }
    
    .feature p {
        margin: 0 !important;
        color: #555 !important;
        font-size: 0.9em !important;
    }
    
    footer {
        text-align: center !important;
        padding: 20px !important;
        color: #666 !important;
        font-size: 0.9em !important;
        border-top: 1px solid #eee !important;
        margin-top: 30px !important;
    }
    
    .loading {
        text-align: center !important;
        padding: 20px !important;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3 !important;
        border-top: 4px solid var(--secondary) !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        animation: spin 1s linear infinite !important;
        margin: 0 auto 10px !important;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg) !important; }
        100% { transform: rotate(360deg) !important; }
    }
    """
    
    # وظيفة المعالجة مع مؤشر التحميل
    def process_with_progress(image, strength):
        """معالجة الصورة مع تحديث التقدم"""
        yield None, "🔄 جاري تحميل النموذج...", None
        
        try:
            restorer = FaceRestorer()
            restorer.load_model()
            yield None, "✅ النموذج جاهز! جاري معالجة الصورة...", None
            
            result, stats = process_face_restoration(image, strength)
            
            if result is None:
                yield None, "❌ فشل في معالجة الصورة", stats
            else:
                yield result, "✅ تمت المعالجة بنجاح!", stats
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            yield None, f"❌ خطأ: {str(e)}", None
    
    # بناء الواجهة
    with gr.Blocks(css=custom_css, title="Ultimate Face Fixer") as demo:
        
        # الرأس
        gr.HTML("""
            <div class="header">
                <h1>✨ Ultimate Face Fixer</h1>
                <p>ترميم وتجميل الصور بتقنية الذكاء الاصطناعي المتطورة</p>
                <div style="margin-top: 10px; font-size: 0.9em;">
                    <span>الإصدار 3.0 | متوافق مع Gradio 6.2.0</span>
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
                    input_image = gr.Image(
                        label="",
                        height=320
                    )
                
                # الصورة الناتجة
                with gr.Column(elem_classes="image-box"):
                    gr.Markdown("### 📥 الصورة المحسنة")
                    output_image = gr.Image(
                        label="",
                        height=320
                    )
            
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
            
            # الميزات
            gr.Markdown("### ✨ المميزات الرئيسية")
            with gr.Row(elem_classes="features"):
                with gr.Column():
                    gr.HTML("""
                        <div class="feature">
                            <h4>🤖 خوارزمية متقدمة</h4>
                            <p>خوارزمية Ultimate Balance الأصلية محفوظة تماماً</p>
                        </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                        <div class="feature">
                            <h4>⚡ معالجة سريعة</h4>
                            <p>دعم كامل لـ GPU/CPU مع معالجة فورية</p>
                        </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                        <div class="feature">
                            <h4>🎯 نتائج دقيقة</h4>
                            <p>ترميم دقيق للملامح مع الحفاظ على التفاصيل</p>
                        </div>
                    """)
                with gr.Column():
                    gr.HTML("""
                        <div class="feature">
                            <h4>📱 واجهة سهلة</h4>
                            <p>واجهة مستخدم بسيطة وسهلة الاستخدام</p>
                        </div>
                    """)
            
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
                
                ### 🛠️ المعلومات التقنية:
                - النموذج: GFPGAN v1.4
                - المكتبات: OpenCV, PyTorch, GFPGAN
                - نظام التشغيل: HuggingFace Spaces
                - الإصدار: Gradio 6.2.0
                """)
            
            # التذييل
            gr.HTML("""
                <footer>
                    <p>Ultimate Face Fixer v3.0 | تم التطوير باستخدام GFPGAN</p>
                    <p style="font-size: 0.8em; color: #888;">
                        ملاحظة: الخوارزمية الأساسية لتحسين الوجه محفوظة تماماً كما هي
                    </p>
                </footer>
            """)
        
        # ربط الأحداث
        process_btn.click(
            fn=process_with_progress,
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

# 7. الدالة الرئيسية
def main():
    """الدالة الرئيسية للتشغيل"""
    print("=" * 60)
    print("Ultimate Face Fixer - الإصدار 3.0")
    print("=" * 60)
    
    # إنشاء الواجهة
    print("🚀 جاري تحميل الواجهة...")
    demo = create_interface()
    
    # تشغيل الواجهة مع إعدادات HuggingFace
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

# 8. إذا كان الملف يعمل كـ __main__
if __name__ == "__main__":
    main()