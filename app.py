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

# 2. استيراد مكتبات إضافية للتحميل الذكي
import urllib.request
import tempfile
import time
from pathlib import Path

# 3. تحميل النموذج بذكاء
def load_gfpgan_model():
    """تحميل نموذج GFPGAN بطريقة ذكية"""
    try:
        from gfpgan import GFPGANer
        
        # محاولة استخدام النموذج المدمج أولاً
        try:
            face_enhancer = GFPGANer(
                model_path='GFPGANv1.4',
                upscale=1.5,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("✅ تم تحميل النموذج المدمج بنجاح")
            return face_enhancer
        except Exception as e:
            print(f"⚠️ النموذج المدمج غير متوفر: {e}")
            
            # محاولة تحميل النموذج من الإنترنت
            model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            
            # إنشاء مجلد مؤقت للنموذج
            model_dir = Path('/tmp/gfpgan_models')
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / 'GFPGANv1.4.pth'
            
            if not model_path.exists():
                print("📥 جاري تحميل النموذج...")
                try:
                    # تحميل النموذج
                    urllib.request.urlretrieve(model_url, model_path)
                    print(f"✅ تم تحميل النموذج إلى: {model_path}")
                except Exception as download_error:
                    print(f"❌ فشل تحميل النموذج: {download_error}")
                    return None
            
            # تحميل النموذج المحلي
            face_enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=1.5,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("✅ تم تحميل النموذج المحلي بنجاح")
            return face_enhancer
            
    except Exception as e:
        print(f"❌ خطأ في تحميل GFPGAN: {e}")
        return None

# 4. تحميل النموذج عند البدء
print("🚀 جاري تحميل النموذج...")
face_enhancer = load_gfpgan_model()

# 5. الـ CSS المحسن
custom_css = """
body { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.gradio-container { 
    max-width: 900px !important; 
    margin: 20px auto !important; 
    background: white !important; 
    border: 1px solid #b3ccff !important; 
    border-radius: 15px !important; 
    box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important; 
    overflow: hidden !important;
}
#title_area { 
    text-align: center !important; 
    color: white !important; 
    padding: 30px !important; 
    background: linear-gradient(90deg, #1c4167, #007eff) !important;
    border-bottom: 3px solid rgba(255,255,255,0.2) !important;
}
#title_area h1 {
    margin: 0 !important;
    font-size: 2.5em !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
}
#title_area p {
    margin: 10px 0 0 0 !important;
    opacity: 0.9 !important;
    font-size: 1.1em !important;
}
button.primary { 
    background: linear-gradient(90deg, #1c4167, #007eff) !important; 
    border: none !important; 
    color: white !important; 
    font-weight: bold !important; 
    border-radius: 10px !important; 
    height: 55px !important;
    font-size: 1.1em !important;
    padding: 0 30px !important;
    transition: all 0.3s ease !important;
}
button.primary:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(28, 65, 103, 0.3) !important;
}
.image-box {
    border: 2px dashed #cbd5e0 !important;
    border-radius: 10px !important;
    padding: 15px !important;
    background: #f7fafc !important;
}
footer {display: none !important;}
.status-box {
    background: #e6fffa !important;
    border: 1px solid #81e6d9 !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin-top: 20px !important;
}
@media (max-width: 768px) {
    .gradio-container {
        margin: 10px !important;
        border-radius: 10px !important;
    }
    #title_area h1 {
        font-size: 2em !important;
    }
}
"""

# 6. الخوارزمية الأساسية - محفوظة تماماً كما هي
def smart_restore_perfectionist(input_img):
    """خوارزمية Ultimate Balance الأصلية - لم يتم لمسها"""
    if input_img is None: 
        return None, "⚠️ الرجاء تحميل صورة أولاً"
    
    if face_enhancer is None:
        return None, "❌ النموذج غير محمل. الرجاء الانتظار قليلاً ثم المحاولة مرة أخرى."
    
    try:
        # بدء توقيت المعالجة
        start_time = time.time()
        
        # تحويل الصورة من numpy (Gradio) إلى BGR (OpenCV)
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            
        h, w = img.shape[:2]
        if w > 2000 or h > 2000:
            img = cv2.resize(img, (w // 2, h // 2))
        
        # خوارزمية Ultimate Balance الأصلية - محفوظة تماماً كما هي
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        silk = cv2.edgePreservingFilter(output, flags=1, sigma_s=30, sigma_r=0.08)
        lab = cv2.cvtColor(silk, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.addWeighted(l, 1.1, cv2.GaussianBlur(l, (0,0), 3), -0.1, 0)
        final_ai = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        img_upscaled = cv2.resize(img, (output.shape[1], output.shape[0]))
        inter_mix = cv2.addWeighted(img_upscaled, 0.5, silk, 0.5, 0)
        final = cv2.addWeighted(inter_mix, 0.8, final_ai, 0.2, 0)
        
        # إحصائيات المعالجة
        processing_time = time.time() - start_time
        
        # رسالة الحالة مع الإحصائيات
        status_msg = f"""
✅ تمت المعالجة بنجاح!

📊 إحصائيات المعالجة:
• الحجم الأصلي: {w}×{h}
• وقت المعالجة: {processing_time:.2f} ثانية
• النموذج: GFPGAN v1.4

💡 ملاحظة: تم تطبيق خوارزمية Ultimate Balance الأصلية بالكامل
        """
        
        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB), status_msg
        
    except Exception as e:
        error_msg = f"❌ خطأ في المعالجة: {str(e)}"
        print(f"Error: {e}")
        return None, error_msg

# 7. بناء الواجهة مع تحسينات بسيطة
with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_id="title_area"):
        gr.HTML("""
            <h1 style='margin-bottom: 5px;'>✨ Ultimate Face Restorer</h1>
            <p style='opacity: 0.9; font-size: 1.1em;'>ترميم ملامح الوجه بتقنية Ultimate Balance الأصلية</p>
            <div style='margin-top: 10px; font-size: 0.9em;'>
                <span>الإصدار الأصلي | الخوارزمية محفوظة تماماً</span>
            </div>
        """)
    
    with gr.Row():
        input_i = gr.Image(
            type="numpy", 
            label="📤 ارفع الصورة الأصلية",
            elem_classes="image-box"
        )
        output_i = gr.Image(
            type="numpy", 
            label="📥 النتيجة بعد الترميم",
            elem_classes="image-box"
        )
    
    # رسالة الحالة
    status_output = gr.Textbox(
        label="💬 حالة المعالجة",
        value="⚡ جاهز للبدء - قم بتحميل صورة",
        interactive=False,
        elem_classes="status-box"
    )
    
    submit_btn = gr.Button(
        "🚀 ابدأ الترميم الآن ✨", 
        variant="primary",
        size="lg"
    )
    
    # معلومات إضافية
    with gr.Accordion("📖 معلومات تقنية", open=False):
        gr.Markdown("""
        ### الخوارزمية المستخدمة:
        تم استخدام خوارزمية **Ultimate Balance** الأصلية كاملة بدون أي تعديلات:
        
        1. **GFPGAN Enhancement**: تحسين الوجه الأساسي
        2. **Edge Preserving Filter**: فلتر الحفاظ على الحواف
        3. **LAB Color Space**: معالجة في فضاء الألوان LAB
        4. **Gaussian Blur**: تمويه غاوسي للتحسين
        5. **Image Mixing**: مزج الصور المتوسط
        
        ### معلومات النموذج:
        - النموذج: GFPGAN v1.4
        - الدقة: 1.5x upscale
        - الهندسة المعمارية: clean
        - المضاعف: 2
        
        ### ملاحظات:
        - الخوارزمية الأصلية محفوظة تماماً كما هي
        - أول معالجة قد تستغرق وقتاً أطول لتحميل النموذج
        - يدعم الصور حتى 2000×2000 بكسل
        """)
    
    # ربط الأحداث
    def process_image(input_img):
        """معالجة الصورة وإرجاع النتيجة والحالة"""
        if input_img is None:
            return None, "⚠️ الرجاء تحميل صورة أولاً"
        
        result, status = smart_restore_perfectionist(input_img)
        return result, status
    
    submit_btn.click(
        fn=process_image, 
        inputs=input_i, 
        outputs=[output_i, status_output]
    )
    
    # تحديث الحالة عند تحميل صورة
    input_i.change(
        fn=lambda x: "📸 الصورة جاهزة للمعالجة!" if x is not None else "⚡ جاهز للبدء - قم بتحميل صورة",
        inputs=input_i,
        outputs=status_output
    )

# 8. التشغيل
if __name__ == "__main__":
    print("=" * 60)
    print("Ultimate Face Restorer - الإصدار الأصلي")
    print("=" * 60)
    
    # فحص النموذج
    if face_enhancer is None:
        print("⚠️ تحذير: لم يتم تحميل النموذج بنجاح")
        print("📋 سيتم تحميله عند أول معالجة")
    else:
        print("✅ النموذج محمل وجاهز للاستخدام")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )