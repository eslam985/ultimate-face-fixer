#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Face Restorer - نسخة HuggingFace Space المتوافقة
إصدار مبسط ومتوافق مع بيئة HuggingFace
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
from PIL import Image
import torch

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 3. تهيئة النموذج (نسخة مبسطة)
class ModelManager:
    """مدير النموذج المبسط"""
    
    def __init__(self):
        self.face_enhancer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def initialize_enhancer(self):
        """تهيئة محسن الوجه"""
        try:
            from gfpgan import GFPGANer
            
            # استخدام النموذج الموجود مسبقاً في HuggingFace
            try:
                # محاولة تحميل النموذج من المسار المحلي
                model_path = '/tmp/GFPGANv1.4.pth'
                if not os.path.exists(model_path):
                    # تحميل النموذج من الإنترنت
                    import gdown
                    model_url = 'https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT'
                    gdown.download(model_url, model_path, quiet=False)
                
                self.face_enhancer = GFPGANer(
                    model_path=model_path,
                    upscale=1.5,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                logger.info("Face enhancer initialized successfully")
                
            except Exception as e:
                logger.warning(f"Could not download model: {e}")
                # استخدام النموذج المدمج في GFPGAN
                self.face_enhancer = GFPGANer(
                    model_path='GFPGANv1.4',
                    upscale=1.5,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize enhancer: {e}")
            raise

# 4. الخوارزمية الأساسية (محفوظة كما هي)
def smart_restore_perfectionist(input_img, strength=1.0):
    """
    الخوارزمية الأساسية - محفوظة تماماً كما هي
    """
    if input_img is None: 
        return None, None
    
    try:
        # التحقق من نوع البيانات
        if isinstance(input_img, dict):
            # Gradio Image component returns dict
            img_array = input_img['image']
        else:
            img_array = input_img
            
        # تحويل الصورة
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # تغيير الحجم إذا لزم الأمر
        h, w = img.shape[:2]
        if w > 2000 or h > 2000:
            img = cv2.resize(img, (w // 2, h // 2))
        
        # تهيئة النموذج
        model_manager = ModelManager()
        model_manager.initialize_enhancer()
        
        # خوارزمية Ultimate Balance الأصلية (غير ملموسة)
        _, _, output = model_manager.face_enhancer.enhance(
            img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )
        
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
        
        # تحسينات إضافية
        final_pil = Image.fromarray(final_rgb)
        
        # إحصائيات المعالجة
        stats = {
            "original_size": f"{h}x{w}",
            "output_size": f"{final.shape[1]}x{final.shape[0]}",
            "strength": strength,
            "processing_time": time.time()
        }
        
        return final_rgb, stats
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None, {"error": str(e)}

# 5. وظائف مساعدة للواجهة
def create_interface():
    """إنشاء واجهة مبسطة ومتوافقة"""
    
    # CSS مبسط
    custom_css = """
    :root {
        --primary-color: #1c4167;
        --secondary-color: #007eff;
        --accent-color: #ff6b6b;
        --background-color: #f9f9f9;
        --card-bg: #ffffff;
    }
    
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    
    .gradio-container {
        max-width: 900px;
        margin: auto;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        padding: 0;
        overflow: hidden;
    }
    
    .header {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 30px 20px;
        text-align: center;
        color: white;
        border-bottom: 5px solid rgba(255,255,255,0.1);
    }
    
    .header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        margin: 10px 0 0;
        opacity: 0.9;
        font-size: 1.1em;
    }
    
    .content {
        padding: 30px;
    }
    
    .image-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    @media (max-width: 768px) {
        .image-section {
            grid-template-columns: 1fr;
        }
    }
    
    .image-box {
        border: 3px dashed #cbd5e0;
        border-radius: 15px;
        padding: 15px;
        background: #f7fafc;
        transition: all 0.3s ease;
        min-height: 400px;
        display: flex;
        flex-direction: column;
    }
    
    .image-box:hover {
        border-color: var(--secondary-color);
        background: #edf2f7;
    }
    
    .controls {
        background: #f8fafc;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 30px;
        border: 1px solid #e2e8f0;
    }
    
    .control-group {
        margin-bottom: 20px;
    }
    
    .control-group label {
        display: block;
        color: var(--primary-color);
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    .strength-slider {
        width: 100%;
    }
    
    .process-btn {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border: none;
        color: white;
        padding: 15px 40px;
        font-size: 1.2em;
        font-weight: 700;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: block;
        width: 100%;
        margin-top: 20px;
    }
    
    .process-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(28, 65, 103, 0.3);
    }
    
    .stats-box {
        background: #e6fffa;
        border: 2px solid #81e6d9;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        font-family: monospace;
    }
    
    .stats-title {
        color: var(--primary-color);
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    .loading {
        text-align: center;
        padding: 40px;
        color: var(--primary-color);
    }
    
    .loading-spinner {
        border: 5px solid #f3f3f3;
        border-top: 5px solid var(--secondary-color);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto 20px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    footer {
        text-align: center;
        padding: 20px;
        color: #718096;
        font-size: 0.9em;
        border-top: 1px solid #e2e8f0;
        margin-top: 30px;
    }
    
    .feature-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    .feature-item {
        background: #f0f9ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid var(--secondary-color);
    }
    """
    
    # وظيفة المعالجة مع مؤشر التحميل
    def process_image_wrapper(input_img, strength):
        """غلاف لوظيفة المعالجة مع إدارة الحالة"""
        if input_img is None:
            return None, None, "⚠️ الرجاء تحميل صورة أولاً"
        
        # عرض مؤشر التحميل
        yield None, None, "🔄 جاري معالجة الصورة... الرجاء الانتظار"
        
        try:
            # معالجة الصورة
            result, stats = smart_restore_perfectionist(input_img, strength)
            
            if result is None:
                yield None, None, "❌ فشل في معالجة الصورة"
            else:
                # تحويل الإحصائيات إلى نص
                stats_text = "📊 إحصائيات المعالجة:\n"
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        if key != "processing_time":
                            stats_text += f"• {key}: {value}\n"
                else:
                    stats_text = str(stats)
                
                yield result, stats_text, "✅ تمت المعالجة بنجاح!"
                
        except Exception as e:
            logger.error(f"Error in wrapper: {e}")
            yield None, None, f"❌ خطأ: {str(e)}"
    
    # بناء الواجهة
    with gr.Blocks(css=custom_css, title="Ultimate Face Fixer") as demo:
        
        # الرأس
        gr.HTML("""
            <div class="header">
                <h1>🎯 Ultimate Face Fixer</h1>
                <p>ترميم وتجميل الصور باستخدام الذكاء الاصطناعي المتطور</p>
                <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                    <span>الإصدار 2.0 | متوافق مع HuggingFace</span>
                </div>
            </div>
        """)
        
        # المحتوى الرئيسي
        with gr.Column(elem_classes="content"):
            
            # قسم الصور
            with gr.Row(elem_classes="image-section"):
                # الصورة المدخلة
                with gr.Column(elem_classes="image-box"):
                    gr.Markdown("### 📤 الصورة الأصلية")
                    input_image = gr.Image(
                        label="",
                        type="numpy",
                        height=350,
                        show_label=False
                    )
                
                # الصورة الناتجة
                with gr.Column(elem_classes="image-box"):
                    gr.Markdown("### 📥 الصورة المحسنة")
                    output_image = gr.Image(
                        label="",
                        type="numpy",
                        height=350,
                        show_label=False
                    )
            
            # عناصر التحكم
            with gr.Column(elem_classes="controls"):
                gr.Markdown("### ⚙️ إعدادات المعالجة")
                
                with gr.Row():
                    with gr.Column():
                        strength_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="🔧 قوة التحسين",
                            info="من خفيف (0.5) إلى قوي (2.0)",
                            elem_classes="strength-slider"
                        )
                    
                    with gr.Column():
                        examples = gr.Examples(
                            examples=[
                                ["https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400"],
                                ["https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400"],
                                ["https://images.unsplash.com/photo-1494790108755-2616b612b786?w-400"]
                            ],
                            inputs=[input_image],
                            label="🖼️ أمثلة سريعة"
                        )
                
                # زر المعالجة
                process_btn = gr.Button(
                    "🚀 بدء الترميم",
                    variant="primary",
                    size="lg",
                    elem_classes="process-btn"
                )
                
                # رسالة الحالة
                status_message = gr.Textbox(
                    label="حالة المعالجة",
                    value="⚡ جاهز للبدء",
                    interactive=False
                )
            
            # الإحصائيات
            with gr.Column(elem_classes="stats-box"):
                gr.Markdown("### 📈 معلومات المعالجة")
                stats_output = gr.Textbox(
                    label="",
                    lines=5,
                    max_lines=10,
                    interactive=False
                )
            
            # الميزات
            with gr.Column():
                gr.Markdown("### ✨ المميزات")
                gr.HTML("""
                    <div class="feature-list">
                        <div class="feature-item">
                            <strong>🤖 خوارزمية متطورة</strong><br>
                            خوارزمية Ultimate Balance الأصلية محفوظة تماماً
                        </div>
                        <div class="feature-item">
                            <strong>⚡ معالجة سريعة</strong><br>
                            دعم كامل للـ GPU والـ CPU
                        </div>
                        <div class="feature-item">
                            <strong>🎯 نتائج دقيقة</strong><br>
                            ترميم وتجميل دقيق للملامح
                        </div>
                        <div class="feature-item">
                            <strong>📱 متوافق تماماً</strong><br>
                            يعمل على HuggingFace Spaces بسلاسة
                        </div>
                    </div>
                """)
            
            # التعليمات
            with gr.Accordion("📖 كيفية الاستخدام", open=False):
                gr.Markdown("""
                ### خطوات الاستخدام:
                1. **ارفع صورة** عن طريق السحب والإفلات أو النقر على منطقة الرفع
                2. **اضبط قوة التحسين** حسب رغبتك (1.0 هي القيمة المثالية)
                3. **انقر على زر "بدء الترميم"**
                4. **انتظر** حتى تظهر النتيجة والإحصائيات
                
                ### ملاحظات هامة:
                - الخوارزمية الأساسية محفوظة تماماً كما هي
                - يدعم معظم صيغ الصور (JPG, PNG, etc.)
                - الحد الأقصى لحجم الصورة: 2000x2000 بكسل
                - المعالجة تستغرق من 5 إلى 30 ثانية حسب حجم الصورة
                
                ### معلومات تقنية:
                - النموذج: GFPGAN v1.4
                - المكتبات: OpenCV, PyTorch, GFPGAN
                - النظام: HuggingFace Spaces
                """)
            
            # التذييل
            gr.HTML("""
                <footer>
                    <p>Ultimate Face Fixer v2.0 | تم التطوير باستخدام GFPGAN وOpenCV</p>
                    <p style="font-size: 0.8em; opacity: 0.7;">
                        تنويه: الخوارزمية الأساسية لتحسين الوجه محفوظة تماماً كما هي
                    </p>
                </footer>
            """)
        
        # ربط الأحداث
        process_btn.click(
            fn=process_image_wrapper,
            inputs=[input_image, strength_slider],
            outputs=[output_image, stats_output, status_message]
        )
        
        # تهيئة تلقائية عند التحميل
        def initialize_on_load():
            try:
                # محاولة تهيئة النموذج عند التحميل
                import threading
                
                def load_model_in_background():
                    try:
                        manager = ModelManager()
                        manager.initialize_enhancer()
                        logger.info("Model loaded successfully in background")
                    except Exception as e:
                        logger.warning(f"Background model loading failed: {e}")
                
                # تحميل النموذج في الخلفية
                threading.Thread(target=load_model_in_background, daemon=True).start()
                
                return "⚡ النظام جاهز للاستخدام!"
            except Exception as e:
                return f"⚠️ Note: {str(e)}"
        
        demo.load(
            fn=initialize_on_load,
            outputs=[status_message]
        )
    
    return demo

# 6. ملف requirements.txt المطلوب لـ HuggingFace
def create_requirements_file():
    """إنشاء ملف المتطلبات"""
    requirements = """torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.8.0
gradio>=4.0.0
numpy>=1.24.0
Pillow>=10.0.0
gfpgan>=1.3.8
realesrgan>=0.3.0
basicsr>=1.4.2
facexlib>=0.3.0
gdown>=4.6.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    logger.info("Requirements file created")

# 7. الدالة الرئيسية
def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("Ultimate Face Fixer - نسخة HuggingFace")
    print("=" * 60)
    
    # إنشاء ملف المتطلبات
    create_requirements_file()
    
    # إنشاء الواجهة
    demo = create_interface()
    
    # تشغيل الواجهة مع إعدادات HuggingFace
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # HuggingFace يدير المشاركة
        debug=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()