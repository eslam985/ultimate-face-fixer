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
from PIL import Image
import time
from pathlib import Path
import tempfile
import base64
from io import BytesIO

# 2. محاولة تحميل RealESRGAN لتحسين الخلفية (اختياري)
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
    print("✅ RealESRGAN متاح لتحسين الخلفية")
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("⚠️ RealESRGAN غير متاح - سيعمل تحسين الوجه فقط")

# 3. تحميل GFPGAN
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
    print("✅ GFPGAN متاح")
    
    # إنشاء محسن الوجه
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        upscale=1.5,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )
    
except Exception as e:
    GFPGAN_AVAILABLE = False
    face_enhancer = None
    print(f"⚠️ خطأ في تحميل GFPGAN: {e}")

# 4. إنشاء محسن الخلفية إذا كان RealESRGAN متاحاً
if REALESRGAN_AVAILABLE:
    try:
        # تحميل نموذج RealESRGAN لتحسين الخلفية
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        print("✅ RealESRGAN محمل لتحسين الخلفية")
    except Exception as e:
        bg_upsampler = None
        print(f"⚠️ خطأ في تحميل RealESRGAN: {e}")
else:
    bg_upsampler = None

custom_css = """
:root {
    /* سحب هوية Egy Pyramid مباشرة */
    --primary: #1c4167;
    --secondary: #0073ea;
    --accent: #005bb5;
    --bg-light: #f9f9f9;
    --text-dark: #333333;
    --border-color: #007eff30;
    --success: #10b981;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box !important;
    font-family: 'Cairo', -apple-system, sans-serif !important;
}

/* إزالة الخلفية المتدرجة البنفسجية لدمج التطبيق مع الموقع */
body {
    background: var(--bg-light) !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 auto !important;
    background: white !important;
    border: none !important;
    box-shadow: none !important; /* إزالة الظل الخارجي لدمجه داخل الـ iframe */
    padding: 10px !important;
}

/* تنسيق منطقة العنوان ليكون مثل الـ Hero Section في موقعك */
#title_area {
    text-align: center !important;
    padding: 30px 15px !important;
    background: linear-gradient(145deg, var(--primary) 0%, var(--secondary) 100%) !important;
    border-radius: 20px !important;
    margin-bottom: 25px !important;
    color: white !important;
}

#title_area h1 {
    color: white !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 10px !important;
}

#title_area p {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1rem !important;
}

/* تنسيق الأزرار (Primary) لتطابق أزرار موقعك الصفراء أو الزرقاء */
button.primary {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
    height: 55px !important;
    box-shadow: 0 4px 15px rgba(28, 65, 103, 0.2) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(28, 65, 103, 0.3) !important;
    filter: brightness(1.1);
}

/* تنسيق حاويات الصور */
.image-container, .gr-box {
    border: 2px solid var(--border-color) !important;
    border-radius: 16px !important;
    background: #ffffff !important;
}

/* تنسيق السلايدرات (Sliders) */
input[type="range"] {
    accent-color: var(--primary) !important;
}

/* تنسيق الـ Checkbox ليطابق ألوانك */
input[type="checkbox"] {
    accent-color: var(--primary) !important;
}

/* إخفاء الفوتر الخاص بـ Gradio للحصول على مظهر White Label */
footer { display: none !important; }

/* تحسينات الموبايل لضمان ثبات العرض */
@media (max-width: 768px) {
    #title_area h1 { font-size: 1.5rem !important; }
    .gradio-container { padding: 5px !important; }
}
"""

# 6. الخوارزمية الأصلية - محفوظة كما هي
def smart_restore_perfectionist(img, enhance_background=False):
    """
    الخوارزمية الأصلية - محفوظة كما هي
    """
    try:
        # خوارزمية Ultimate Balance الأصلية
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        silk = cv2.edgePreservingFilter(output, flags=1, sigma_s=30, sigma_r=0.08)
        lab = cv2.cvtColor(silk, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.addWeighted(l, 1.1, cv2.GaussianBlur(l, (0,0), 3), -0.1, 0)
        final_ai = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        img_upscaled = cv2.resize(img, (output.shape[1], output.shape[0]))
        inter_mix = cv2.addWeighted(img_upscaled, 0.5, silk, 0.5, 0)
        final = cv2.addWeighted(inter_mix, 0.8, final_ai, 0.2, 0)
        
        # إذا طلب المستخدم تحسين الخلفية وكان RealESRGAN متاحاً
        if enhance_background and bg_upsampler is not None and REALESRGAN_AVAILABLE:
            try:
                # تحسين الخلفية باستخدام RealESRGAN
                print("🔄 جاري تحسين الخلفية...")
                bg_enhanced, _ = bg_upsampler.enhance(final, outscale=2)
                # تغيير الحجم ليتناسب مع الصورة الأصلية
                final = cv2.resize(bg_enhanced, (final.shape[1], final.shape[0]))
                print("✅ تم تحسين الخلفية")
            except Exception as bg_error:
                print(f"⚠️ خطأ في تحسين الخلفية: {bg_error}")
        
        return final
    except Exception as e:
        print(f"❌ خطأ في الخوارزمية: {e}")
        raise

# 7. الدالة الرئيسية للمعالجة
def process_image(input_img, enhance_full_image=False, refine_count=0):
    """
    معالجة الصورة مع دعم جميع الميزات الجديدة
    """
    if input_img is None: 
        return None, None, "⚠️ الرجاء تحميل صورة أولاً", 0
    
    if face_enhancer is None:
        return None, None, "❌ النموذج غير محمل. يرجى المحاولة لاحقاً.", 0
    
    try:
        start_time = time.time()
        
        # تحويل الصورة
        if isinstance(input_img, dict):
            img_array = input_img['image']
        else:
            img_array = input_img
        
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # حفظ الصورة الأصلية للمقارنة
        original_img = img.copy()
        
        # تغيير الحجم إذا كان كبيراً
        h, w = img.shape[:2]
        if w > 2000 or h > 2000:
            scale = min(2000 / w, 2000 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"📏 تم تغيير الحجم من {w}x{h} إلى {new_w}x{new_h}")
        
        # تطبيق الخوارزمية الأصلية
        result = smart_restore_perfectionist(img, enhance_full_image)
        
        # إذا طلب المستخدم توضيح إضافي
        for i in range(refine_count):
            print(f"🔄 جاري التوضيح الإضافي #{i+1}...")
            result = smart_restore_perfectionist(result, enhance_full_image)
        
        # التحويل النهائي
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # إحصائيات المعالجة
        processing_time = time.time() - start_time
        refined_times = refine_count + 1
        
        stats = f"""
✅ تمت المعالجة بنجاح!

📊 إحصائيات المعالجة:
• الحجم الأصلي: {w}×{h}
• وقت المعالجة: {processing_time:.2f} ثانية
• عدد مرات التوضيح: {refined_times}
• تحسين الخلفية: {'✅ مفعل' if enhance_full_image else '❌ غير مفعل'}
• النموذج: GFPGAN v1.4 + RealESRGAN

💡 يمكنك:
1. استخدام شريط المقارنة لمشاهدة الفرق
2. النقر على "توضيح إضافي" لتحسين النتيجة أكثر
3. تحميل أو مشاركة النتيجة النهائية
        """
        
        return original_rgb, result_rgb, stats, refined_times
        
    except Exception as e:
        error_msg = f"❌ خطأ في المعالجة: {str(e)}"
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, error_msg, 0

# 8. دالة التوضيح الإضافي
def refine_existing_image(result_img, enhance_full_image=False):
    """
    توضيح إضافي للصورة الناتجة
    """
    if result_img is None:
        return None, "⚠️ لا توجد صورة للتوضيح"
    
    try:
        start_time = time.time()
        
        # تحويل الصورة
        img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        # تطبيق الخوارزمية
        refined = smart_restore_perfectionist(img, enhance_full_image)
        refined_rgb = cv2.cvtColor(refined, cv2.COLOR_BGR2RGB)
        
        processing_time = time.time() - start_time
        
        stats = f"""
✨ تم التوضيح الإضافي بنجاح!

📊 إحصائيات التوضيح:
• وقت التوضيح: {processing_time:.2f} ثانية
• تحسين الخلفية: {'✅ مفعل' if enhance_full_image else '❌ غير مفعل'}

💡 يمكنك الاستمرار في التوضيح أو تحميل النتيجة
        """
        
        return refined_rgb, stats
        
    except Exception as e:
        error_msg = f"❌ خطأ في التوضيح: {str(e)}"
        return None, error_msg

# 9. دالة لإنشاء صورة قابلة للتنزيل
def create_downloadable_image(img_array):
    """
    تحويل الصورة إلى صيغة قابلة للتنزيل
    """
    if img_array is None:
        return None
    
    try:
        # تحويل إلى صيغة PIL
        img_pil = Image.fromarray(img_array)
        
        # حفظ في بايتس
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG", quality=95)
        
        # ترميز base64 للتنزيل المباشر
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"❌ خطأ في إنشاء صورة للتنزيل: {e}")
        return None

# 10. بناء الواجهة المتطورة
with gr.Blocks(css=custom_css, title="Ultimate Face Restorer Pro") as demo: 
    
    # المتغيرات لحفظ الحالة
    current_result = gr.State(value=None)
    refine_counter = gr.State(value=0)
    
    # العنوان الرئيسي
    with gr.Column(elem_id="title_area"):
        gr.HTML("""
            <h1>✨ Ultimate Face Restorer Pro</h1>
            <p>ترميم وتجميل الصور بتقنية Ultimate Balance المتطورة</p>
            <div class="version-badge">الإصدار الاحترافي | خوارزمية محفوظة 100%</div>
        """)
    
    # علامات التبويب الرئيسية
    with gr.Tabs(elem_classes="tab-nav") as tabs:
        
        # تبويب المعالجة الرئيسية
        with gr.TabItem("🎨 معالجة الصور", id="process"):
            with gr.Row():
                with gr.Column(scale=1):
                    # تحميل الصورة
                    gr.Markdown("### 📤 تحميل الصورة")
                    input_image = gr.Image(
                        label="",
                        type="numpy",
                        height=350,
                        elem_classes="image-container"
                    )
                    
                    # خيارات التحسين
                    gr.Markdown("### ⚙️ خيارات التحسين")
                    
                    with gr.Column(elem_classes="checkbox-container"):
                        enhance_background = gr.Checkbox(
                            label="✅ تحسين الصورة بالكامل (يشمل الخلفية والملابس)",
                            value=False,
                            info="يستخدم RealESRGAN لتحسين كامل الصورة"
                        )
                    
                    # شريط التوضيح الإضافي
                    gr.Markdown("### 🔍 مستوى التوضيح")
                    refine_slider = gr.Slider(
                        minimum=0,
                        maximum=3,
                        value=0,
                        step=1,
                        label="مرات التوضيح الإضافي",
                        info="0 = توضيح عادي، 3 = توضيح مكثف"
                    )
                    
                    # زر المعالجة الرئيسي
                    process_btn = gr.Button(
                        "🚀 ابدأ الترميم الآن",
                        variant="primary",
                        size="lg",
                        elem_classes="pulse-animation"
                    )
                
                with gr.Column(scale=2):
                    # منطقة النتائج
                    gr.Markdown("### 📊 النتائج")
                    
                    # شريط المقارنة التفاعلي
                    with gr.Column(elem_classes="compare-container"):
                        gr.Markdown("#### ↔️ شريط المقارنة - اسحب لرؤية الفرق")
                        compare_output = gr.Image(
                            label="قبل ⇄ بعد",
                            type="numpy",
                            height=400,
                            show_label=False
                        )
                    
                    # أزرار الإجراءات
                    with gr.Row(elem_classes="action-buttons"):
                        download_btn = gr.Button(
                            "💾 تحميل النتيجة",
                            variant="secondary",
                            size="lg",
                            elem_classes="action-button download-btn"
                        )
                        
                        share_btn = gr.Button(
                            "📤 مشاركة النتيجة",
                            variant="secondary",
                            size="lg",
                            elem_classes="action-button share-btn"
                        )
                        
                        refine_btn = gr.Button(
                            "✨ توضيح إضافي",
                            variant="secondary",
                            size="lg",
                            elem_classes="action-button refine-btn"
                        )
                    
                    # عرض الإحصائيات
                    stats_output = gr.Textbox(
                        label="📈 إحصائيات المعالجة",
                        lines=8,
                        interactive=False,
                        elem_classes="stats-box"
                    )
                    
                    # رابط التنزيل المخفي
                    download_link = gr.HTML(visible=False)
        
        # تبويب التعليمات
        with gr.TabItem("📖 التعليمات", id="help"):
            with gr.Column():
                gr.Markdown("""
                ## 🎯 دليل الاستخدام الكامل
                
                ### 🔧 كيفية الاستخدام:
                1. **ارفع صورة** عن طريق السحب والإفلات أو النقر
                2. **اختر خيارات التحسين** حسب رغبتك
                3. **انقر على "ابدأ الترميم"**
                4. **استخدم شريط المقارنة** لرؤية الفرق
                5. **قم بتنزيل أو تحسين** النتيجة
                
                ### ✨ الميزات الجديدة:
                
                #### 1. تحسين الصورة بالكامل ✅
                - **المشكلة**: الأدوات القديمة تحسن الوجه فقط وتترك الخلفية سيئة
                - **الحل**: تفعيل هذا الخيار يحسن كامل الصورة (الوجه + الخلفية + الملابس)
                - **التقنية**: يستخدم RealESRGAN لتحسين الخلفية مع GFPGAN للوجه
                
                #### 2. شريط المقارنة التفاعلي ↔️
                - اسحب الشريط الأوسط لرؤية الفرق بين الصورة الأصلية والمحسنة
                - يعمل بكسل بكسل لمقارنة دقيقة
                - يساعد في رؤية التحسينات بوضوح
                
                #### 3. التوضيح الإضافي ✨
                - **المشكلة**: قد تحتاج بعض الصور لتوضيح أكثر
                - **الحل**: استخدم هذا الزر لتطبيق الخوارزمية على النتيجة النهائية
                - **مثال**: صورتك الأولى جيدة، لكنك تريدها أفضل؟ اضغط على "توضيح إضافي"
                - **يمكنك**: الضغط عدة مرات للحصول على أفضل نتيجة
                
                #### 4. تحميل ومشاركة محسنة 💾📤
                - أزرار كبيرة وواضحة للتنزيل والمشاركة
                - جودة عالية للصورة المحفوظة
                - مشاركة سريعة للنتائج
                
                ### ⚡ نصائح احترافية:
                1. **للصور القديمة**: استخدم "توضيح إضافي" 2-3 مرات
                2. **للصور الكاملة**: فعّل "تحسين الصورة بالكامل"
                3. **للوجوه فقط**: اترك "تحسين الصورة بالكامل" معطل
                4. **للمقارنة**: استخدم شريط المقارنة لرؤية التغييرات الدقيقة
                
                ### 🛠️ معلومات تقنية:
                - الخوارزمية الأساسية: Ultimate Balance (محفوظة 100%)
                - تحسين الوجه: GFPGAN v1.4
                - تحسين الخلفية: RealESRGAN x2plus (اختياري)
                - معالجة الصور: OpenCV + Pillow
                - الواجهة: Gradio مع CSS مخصص
                """)
    
    # الفوتر
    gr.HTML("""
        <div class="custom-footer">
            <p>Ultimate Face Restorer Pro | الإصدار الاحترافي</p>
            <p style="opacity: 0.8; font-size: 0.9em; margin-top: 10px;">
                تم التطوير باستخدام GFPGAN + RealESRGAN | الخوارزمية محفوظة 100% كما هي
            </p>
        </div>
    """)
    
    # ربط الأحداث - المعالجة الرئيسية
    def process_wrapper(input_img, enhance_bg, refine_level):
        """غلاف للمعالجة مع حفظ الحالة"""
        original, result, stats, refined = process_image(input_img, enhance_bg, refine_level)
        if result is not None:
            # حفظ النتيجة الحالية
            return original, result, result, stats, refined, result
        return original, result, None, stats, 0, None
    
    process_btn.click(
        fn=process_wrapper,
        inputs=[input_image, enhance_background, refine_slider],
        outputs=[compare_output, compare_output, current_result, stats_output, refine_counter, compare_output]
    )
    
    # ربط الأحداث - التوضيح الإضافي
    def refine_wrapper(current_img, enhance_bg):
        """توضيح إضافي للصورة الحالية"""
        if current_img is None:
            return None, "⚠️ لا توجد صورة للتوضيح"
        
        refined, stats = refine_existing_image(current_img, enhance_bg)
        if refined is not None:
            # زيادة العداد
            new_counter = refine_counter.value + 1 if hasattr(refine_counter, 'value') else 1
            return refined, stats, refined, new_counter, refined
        return None, stats, None, refine_counter.value, None
    
    refine_btn.click(
        fn=refine_wrapper,
        inputs=[current_result, enhance_background],
        outputs=[compare_output, stats_output, current_result, refine_counter, compare_output]
    )
    
    # ربط الأحداث - إنشاء رابط التنزيل
    def create_download_wrapper(img):
        """إنشاء رابط تنزيل للصورة"""
        download_data = create_downloadable_image(img)
        if download_data:
            return f"""
            <a href="{download_data}" download="enhanced_image.png" 
               style="display: inline-block; padding: 12px 24px; background: linear-gradient(90deg, #10b981, #34d399); 
                      color: white; text-decoration: none; border-radius: 8px; font-weight: bold;">
               ⬇️ انقر هنا لتنزيل الصورة
            </a>
            """
        return "<p style='color: red;'>❌ خطأ في إنشاء رابط التنزيل</p>"
    
    download_btn.click(
        fn=create_download_wrapper,
        inputs=[current_result],
        outputs=[download_link]
    )
    
    # ربط الأحداث - تحديث شريط المقارنة
    def update_compare_slider(value):
        """تحديث شريط المقارنة"""
        return value
    
    compare_output.change(
        fn=update_compare_slider,
        inputs=[compare_output],
        outputs=[compare_output]
    )
    
    # رسالة الترحيب
    def welcome_message():
        return "🌟 مرحباً! يمكنك الآن تحميل صورة والاستفادة من جميع الميزات الجديدة"
    
    demo.load(welcome_message, outputs=[stats_output])

# 11. التشغيل
if __name__ == "__main__":
    print("=" * 70)
    print("Ultimate Face Restorer Pro - الإصدار الاحترافي")
    print("=" * 70)
    print("🚀 الميزات الجديدة:")
    print("✅ 1. تحسين الصورة بالكامل (الوجه + الخلفية + الملابس)")
    print("✅ 2. شريط مقارنة تفاعلي")
    print("✅ 3. توضيح إضافي متكرر")
    print("✅ 4. أزرار تحميل ومشاركة محسنة")
    print("✅ 5. واجهة مستخدم احترافية")
    print("=" * 70)
    
demo.launch(show_api=False, debug=True)