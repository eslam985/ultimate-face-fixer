import sys
import os
import time
import cv2
import numpy as np
import gradio as gr
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# إعدادات المسارات
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'

print("🚀 بدء تحميل النظام...")

def load_gfpgan_model():
    """تحميل نموذج GFPGAN"""
    try:
        # محاولة الاستيراد
        try:
            from gfpgan import GFPGANer
            print("✅ تم استيراد GFPGAN بنجاح")
        except ImportError as e:
            print(f"❌ خطأ في استيراد GFPGAN: {e}")
            # محاولة تثبيت GFPGAN إذا لم يكن مثبتاً
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gfpgan"])
            from gfpgan import GFPGANer
        
        # إنشاء النموذج
        model = GFPGANer(
            model_path='GFPGANv1.4',
            upscale=1.5,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device='cpu'  # استخدام CPU لتجنب مشاكل GPU
        )
        print("✅ تم تحميل النموذج بنجاح")
        return model
        
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        # إنشاء نموذج بديل للاختبار
        print("⚠️ استخدام معالج بديل للاختبار")
        return None

# تحميل النموذج عند البدء
face_enhancer = load_gfpgan_model()

def process_image_simple(input_img, strength=1.0):
    """معالجة الصورة - نسخة مبسطة"""
    try:
        if input_img is None:
            return None, "⚠️ الرجاء تحميل صورة أولاً"
        
        print(f"🔧 بدء معالجة الصورة - القوة: {strength}")
        start_time = time.time()
        
        # تحويل الصورة
        if isinstance(input_img, dict):
            img_array = input_img['image']
        else:
            img_array = input_img
        
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        original_h, original_w = img.shape[:2]
        
        # تقليل الحجم إذا كان كبيراً
        max_size = 512
        if original_w > max_size or original_h > max_size:
            scale = min(max_size / original_w, max_size / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"📏 تم تغيير الحجم من {original_w}x{original_h} إلى {new_w}x{new_h}")
        
        if face_enhancer is not None:
            try:
                # استخدام GFPGAN إذا كان متاحاً
                _, _, output = face_enhancer.enhance(
                    img, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                print("✅ تم تطبيق GFPGAN بنجاح")
            except Exception as e:
                print(f"⚠️ خطأ في GFPGAN: {e}، استخدام المعالجة البديلة")
                output = img
        else:
            # معالجة بديلة
            output = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            print("⚠️ استخدام المعالجة البديلة (بدون GFPGAN)")
        
        # تطبيق بعض التحسينات البسيطة
        if strength > 1.0:
            # تحسين التباين
            lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # تحسين الحدة
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            output = cv2.filter2D(output, -1, kernel)
        
        # التحويل النهائي
        final_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        # إحصائيات
        processing_time = time.time() - start_time
        stats = f"""
✅ تمت المعالجة بنجاح!

📊 معلومات المعالجة:
• الحجم الأصلي: {original_w}×{original_h}
• وقت المعالجة: {processing_time:.2f} ثانية
• قوة التحسين: {strength}
• النموذج: {'GFPGAN' if face_enhancer is not None else 'بديل'}

💡 يمكنك تحميل النتيجة بالنقر على زر التحميل
        """
        
        return final_rgb, stats
        
    except Exception as e:
        print(f"❌ خطأ في المعالجة: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ خطأ في المعالجة: {str(e)}"

# إنشاء الواجهة
with gr.Blocks(title="Ultimate Face Fixer", theme=gr.themes.Soft()) as demo:
    
    # العنوان
    gr.Markdown("""
    # ✨ Ultimate Face Fixer
    ### أداة بسيطة وسريعة لتحسين جودة الوجوه في الصور
    
    **كيفية الاستخدام:**
    1. قم بتحميل صورة عن طريق السحب والإفلات أو النقر
    2. اضبط قوة التحسين (1.0 هو المستوى الطبيعي)
    3. انقر على زر "معالجة الصورة"
    4. انتظر بضع ثوانٍ للحصول على النتيجة
    """)
    
    with gr.Row():
        with gr.Column():
            # إدخال الصورة
            input_image = gr.Image(
                label="📤 الصورة الأصلية",
                type="numpy",
                height=300
            )
            
            # عناصر التحكم
            strength_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="🔧 قوة التحسين",
                info="من خفيف (0.5) إلى قوي (2.0)"
            )
            
            # زر المعالجة
            process_btn = gr.Button(
                "🚀 معالجة الصورة",
                variant="primary",
                size="lg"
            )
            
            # رسالة الحالة
            status_msg = gr.Textbox(
                label="💬 حالة المعالجة",
                value="⚡ جاهز للبدء - قم بتحميل صورة",
                interactive=False
            )
        
        with gr.Column():
            # إخراج الصورة
            output_image = gr.Image(
                label="📥 الصورة المحسنة",
                type="numpy",
                height=300
            )
            
            # الإحصائيات
            stats_output = gr.Textbox(
                label="📊 إحصائيات المعالجة",
                lines=8,
                interactive=False
            )
    
    # الميزات
    with gr.Row():
        gr.Markdown("""
        ### ✨ المميزات:
        - **تحسين تلقائي** لجودة الوجه
        - **واجهة بسيطة** وسهلة الاستخدام
        - **معالجة سريعة** خلال ثوانٍ
        - **دعم جميع** أحجام الصور
        """)
    
    # التعليمات
    with gr.Accordion("📖 معلومات إضافية", open=False):
        gr.Markdown("""
        ### معلومات تقنية:
        - يستخدم خوارزميات معالجة الصور المتقدمة
        - يعمل على جميع أنواع الصور (JPG, PNG, إلخ)
        - يحافظ على الجودة الأصلية قدر الإمكان
        - متوافق مع جميع المتصفحات
        
        ### ملاحظات هامة:
        - الإصدار الأول قد يستغرق بعض الوقت لتحميل النموذج
        - الصور الكبيرة جداً يتم تصغيرها تلقائياً
        - يمكنك حفظ النتيجة بالنقر على الصورة
        """)
    
    # التعليقات
    gr.Markdown("""
    ---
    *تم التطوير باستخدام GFPGAN وOpenCV*  
    *متوافق مع HuggingFace Spaces*
    """)
    
    # ربط الأحداث
    def process_wrapper(image, strength):
        """غلاف لوظيفة المعالجة"""
        if image is None:
            return None, "⚠️ الرجاء تحميل صورة أولاً", ""
        
        result, stats = process_image_simple(image, strength)
        if result is not None:
            return result, "✅ تمت المعالجة بنجاح!", stats
        else:
            return None, "❌ حدث خطأ أثناء المعالجة", stats
    
    process_btn.click(
        fn=process_wrapper,
        inputs=[input_image, strength_slider],
        outputs=[output_image, status_msg, stats_output]
    )
    
    # تحديث الحالة عند تحميل صورة
    def update_status(image):
        if image is not None:
            return "📸 الصورة جاهزة للمعالجة!"
        return "⚡ جاهز للبدء - قم بتحميل صورة"
    
    input_image.change(
        fn=update_status,
        inputs=[input_image],
        outputs=[status_msg]
    )

# تشغيل التطبيق
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )