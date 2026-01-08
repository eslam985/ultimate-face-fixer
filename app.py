#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Face Restorer - النسخة الاحترافية
أداة متقدمة لترميم وتجميل الصور مع الحفاظ على خوارزمية توضيح الوجه الأصلية
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
import json

# 1. الإعدادات الأولية والإصلاحات
import warnings
warnings.filterwarnings('ignore')

# حقنة الإصلاح الإجبارية
import torchvision
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

# 2. استيراد المكتبات
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageOps, ImageFilter
import torch
import hashlib
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# 3. إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 4. إعدادات المحرك والتهيئة المتقدمة
class Config:
    """فئة التهيئة المتقدمة"""
    VERSION = "2.0.0"
    MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    MODEL_CACHE_DIR = Path.home() / '.cache' / 'ultimate_face_restorer'
    OUTPUT_DIR = Path.home() / 'UltimateFaceRestorer' / 'results'
    MAX_IMAGE_SIZE = 4000
    MIN_IMAGE_SIZE = 256
    DEFAULT_UPSCALE = 1.5
    BATCH_SIZE = 4
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    
    @staticmethod
    def setup_directories():
        """إنشاء المجلدات المطلوبة"""
        Config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# إعداد المجلدات
Config.setup_directories()

# 5. إدارة النماذج المتقدمة
class ModelManager:
    """مدير النماذج الذكي"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.face_enhancer = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._initialized = True
            logger.info(f"Using device: {self.device}")
    
    def initialize_enhancer(self):
        """تهيئة محسن الوجه"""
        try:
            from gfpgan import GFPGANer
            
            model_path = Config.MODEL_CACHE_DIR / 'GFPGANv1.4.pth'
            
            # تحميل النموذج إذا لم يكن موجوداً
            if not model_path.exists():
                logger.info("Downloading model...")
                import urllib.request
                urllib.request.urlretrieve(Config.MODEL_URL, model_path)
                logger.info("Model downloaded successfully")
            
            self.face_enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=Config.DEFAULT_UPSCALE,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            logger.info("Face enhancer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhancer: {e}")
            raise
    
    def get_enhancer(self):
        """الحصول على محسن الوجه"""
        if self.face_enhancer is None:
            self.initialize_enhancer()
        return self.face_enhancer

# 6. معالجة الصور المتقدمة
class ImageProcessor:
    """معالج الصور المتقدم"""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """التحقق من صحة الصورة"""
        if image is None:
            return False
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False
        if image.size == 0:
            return False
        return True
    
    @staticmethod
    def smart_resize(image: np.ndarray, max_size: int = 2000) -> np.ndarray:
        """تغيير حجم الصورة بذكاء"""
        h, w = image.shape[:2]
        
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    @staticmethod
    def enhance_quality(image: np.ndarray) -> np.ndarray:
        """تحسين جودة الصورة الأساسية"""
        # تحسين التباين باستخدام CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # تخفيف الضوضاء
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    @staticmethod
    def generate_metadata(image_hash: str, processing_time: float) -> Dict:
        """إنشاء بيانات وصفية للصورة"""
        return {
            "image_hash": image_hash,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "version": Config.VERSION,
            "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        }

# 7. الخوارزمية الرئيسية (محفوظة كما هي)
def smart_restore_perfectionist(
    input_img: np.ndarray,
    enhance_preprocess: bool = True,
    strength: float = 1.0
) -> Optional[np.ndarray]:
    """
    الخوارزمية الرئيسية لترميم الوجه - محفوظة كما هي
    """
    if input_img is None: 
        return None
    
    try:
        # التحقق من الصورة
        if not ImageProcessor.validate_image(input_img):
            logger.error("Invalid image format")
            return None
        
        # تحويل الصورة
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        
        # تغيير الحجم إذا لزم الأمر
        img = ImageProcessor.smart_resize(img, Config.MAX_IMAGE_SIZE)
        
        # تحسين الجودة المبدئي
        if enhance_preprocess:
            img = ImageProcessor.enhance_quality(img)
        
        # خوارزمية Ultimate Balance الأصلية (غير ملموسة)
        model_manager = ModelManager()
        face_enhancer = model_manager.get_enhancer()
        
        _, _, output = face_enhancer.enhance(
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
        
        # تحسين النهائي
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        # تحسين الجودة النهائية
        final = Image.fromarray(final)
        final = final.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        return np.array(final)
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None

# 8. المعالجة الدفعية
def batch_process_images(
    images: List[np.ndarray],
    progress_callback=None
) -> List[Optional[np.ndarray]]:
    """معالجة عدة صور دفعة واحدة"""
    results = []
    
    with ThreadPoolExecutor(max_workers=Config.BATCH_SIZE) as executor:
        futures = []
        for img in images:
            future = executor.submit(smart_restore_perfectionist, img)
            futures.append(future)
            
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                
                if progress_callback:
                    progress = (i + 1) / len(images)
                    progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.append(None)
    
    return results

# 9. إدارة الملفات المتقدمة
class FileManager:
    """مدير الملفات المتقدم"""
    
    @staticmethod
    def generate_filename(original_name: str) -> str:
        """إنشاء اسم ملف فريد"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(original_name.encode()).hexdigest()[:8]
        return f"restored_{timestamp}_{name_hash}.png"
    
    @staticmethod
    def save_image(image: np.ndarray, filename: str) -> str:
        """حفظ الصورة بجودة عالية"""
        output_path = Config.OUTPUT_DIR / filename
        
        # استخدام جودة عالية للصورة
        img_pil = Image.fromarray(image)
        img_pil.save(output_path, 'PNG', optimize=True, quality=95)
        
        return str(output_path)
    
    @staticmethod
    def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
        """تحميل الصور من مجلد"""
        images = []
        folder = Path(folder_path)
        
        if not folder.exists():
            return images
        
        for ext in Config.SUPPORTED_FORMATS:
            for file_path in folder.glob(f"*{ext}"):
                try:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return images

# 10. الواجهة المتقدمة
class AdvancedInterface:
    """فئة الواجهة المتقدمة"""
    
    @staticmethod
    def create_interface():
        """إنشاء واجهة مستخدم متقدمة"""
        
        # CSS متقدم
        custom_css = """
        :root {
            --primary-color: #1c4167;
            --secondary-color: #007eff;
            --accent-color: #ff6b6b;
            --background-color: #f9f9f9;
            --card-bg: #ffffff;
            --text-primary: #2d3748;
            --text-secondary: #718096;
            --border-radius: 12px;
            --shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        
        .gradio-container {
            max-width: 1200px !important;
            margin: 2rem auto !important;
            background: var(--card-bg) !important;
            border-radius: var(--border-radius) !important;
            box-shadow: var(--shadow) !important;
            border: none !important;
            padding: 20px !important;
        }
        
        #title_area {
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            margin-bottom: 30px;
        }
        
        #title_area h1 {
            color: white;
            font-size: 2.8em;
            margin-bottom: 10px;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        #title_area p {
            color: rgba(255,255,255,0.9);
            font-size: 1.2em;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .control-panel {
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            border-radius: var(--border-radius);
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        .control-title {
            color: var(--primary-color);
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .control-title i {
            font-size: 1.2em;
        }
        
        .image-container {
            border: 2px dashed #cbd5e0;
            border-radius: var(--border-radius);
            padding: 15px;
            background: #f7fafc;
            transition: all 0.3s ease;
        }
        
        .image-container:hover {
            border-color: var(--secondary-color);
            background: #edf2f7;
        }
        
        button.primary {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
            border: none !important;
            color: white !important;
            font-weight: 700 !important;
            border-radius: var(--border-radius) !important;
            height: 55px !important;
            font-size: 1.1em !important;
            padding: 0 40px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(28, 65, 103, 0.3) !important;
        }
        
        button.primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(28, 65, 103, 0.4) !important;
        }
        
        .progress-bar {
            height: 8px !important;
            border-radius: 4px !important;
            background: linear-gradient(90deg, #4fd1c7, #38b2ac) !important;
        }
        
        .stats-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: var(--border-radius);
            padding: 20px;
            margin-top: 20px;
        }
        
        .stats-title {
            color: var(--primary-color);
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .stats-value {
            color: var(--text-primary);
            font-size: 1.3em;
            font-weight: 700;
        }
        
        .model-info {
            background: linear-gradient(145deg, #e6fffa, #b2f5ea);
            border: 1px solid #81e6d9;
            border-radius: var(--border-radius);
            padding: 15px;
            margin-top: 15px;
        }
        
        .tab-nav {
            border-radius: var(--border-radius) !important;
            overflow: hidden !important;
            background: #edf2f7 !important;
        }
        
        .tab-nav button {
            border-radius: 0 !important;
            font-weight: 600 !important;
        }
        
        .compare-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        @media (max-width: 768px) {
            .gradio-container {
                margin: 1rem !important;
                padding: 15px !important;
            }
            
            #title_area h1 {
                font-size: 2em;
            }
            
            .compare-container {
                grid-template-columns: 1fr;
            }
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: var(--text-secondary);
            font-size: 0.9em;
        }
        
        .feature-icon {
            background: var(--primary-color);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            margin-right: 15px;
        }
        
        .feature-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow);
        }
        """
        
        # JavaScript مخصص
        custom_js = """
        function updateStats() {
            const timestamp = new Date().toLocaleString();
            const version = "2.0.0";
            const device = navigator.hardwareConcurrency ? `CPU Cores: ${navigator.hardwareConcurrency}` : "Device info unavailable";
            
            return {
                timestamp: timestamp,
                version: version,
                device: device,
                userAgent: navigator.userAgent
            };
        }
        """
        
        # وظائف المساعدة
        def process_single_image(input_img, strength, enhance_preprocess):
            """معالجة صورة واحدة"""
            start_time = time.time()
            
            result = smart_restore_perfectionist(
                input_img,
                enhance_preprocess=enhance_preprocess,
                strength=strength
            )
            
            processing_time = time.time() - start_time
            
            if result is not None:
                # حفظ الصورة
                filename = FileManager.generate_filename("single_image")
                save_path = FileManager.save_image(result, filename)
                
                stats = {
                    "processing_time": f"{processing_time:.2f} seconds",
                    "output_size": f"{result.shape[1]}x{result.shape[0]}",
                    "output_path": save_path,
                    "success": True
                }
            else:
                stats = {
                    "processing_time": f"{processing_time:.2f} seconds",
                    "error": "Failed to process image",
                    "success": False
                }
            
            return result, json.dumps(stats, indent=2)
        
        def process_batch_images(folder_path, strength, progress=gr.Progress()):
            """معالجة مجموعة صور"""
            if not folder_path:
                return [], "Please select a folder"
            
            images = FileManager.load_images_from_folder(folder_path)
            
            if not images:
                return [], "No valid images found in the folder"
            
            results = []
            processed_count = 0
            
            def progress_callback(p):
                progress((processed_count + p) / len(images), f"Processing image {processed_count + 1}/{len(images)}")
            
            for i, img in enumerate(images):
                try:
                    result = smart_restore_perfectionist(
                        img,
                        enhance_preprocess=True,
                        strength=strength
                    )
                    
                    if result is not None:
                        filename = FileManager.generate_filename(f"batch_{i}")
                        save_path = FileManager.save_image(result, filename)
                        results.append(result)
                        processed_count += 1
                        
                        progress_callback(1)
                        
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
            
            stats = {
                "total_images": len(images),
                "processed_successfully": processed_count,
                "failed": len(images) - processed_count,
                "output_folder": str(Config.OUTPUT_DIR)
            }
            
            return results[:4], json.dumps(stats, indent=2)  # إرجاع أول 4 صور للمعاينة
        
        # بناء الواجهة
        with gr.Blocks(css=custom_css, js=custom_js, title="Ultimate Face Restorer Pro") as demo:
            
            # رأس الصفحة
            with gr.Column(elem_id="title_area"):
                gr.HTML("""
                    <div style="text-align: center;">
                        <h1>🔄 Ultimate Face Restorer Pro</h1>
                        <p style="font-size: 1.2em; opacity: 0.9;">ترميم وتجميل الصور بتقنية الذكاء الاصطناعي المتطورة - الإصدار الاحترافي</p>
                    </div>
                """)
            
            # علامات التبويب الرئيسية
            with gr.Tabs() as tabs:
                
                # علامة التبويب: المعالجة الفردية
                with gr.TabItem("🎨 معالجة فردية", id="single"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # لوحة التحكم
                            with gr.Column(scale=1, elem_classes="control-panel"):
                                gr.Markdown("### ⚙️ إعدادات المعالجة")
                                
                                strength_slider = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="قوة التحسين",
                                    info="قوة التحسين: من خفيف (0.5) إلى قوي (2.0)"
                                )
                                
                                enhance_preprocess = gr.Checkbox(
                                    label="تحسين مسبق للجودة",
                                    value=True,
                                    info="تحسين جودة الصورة قبل المعالجة"
                                )
                                
                                gr.Markdown("---")
                                
                                process_btn = gr.Button(
                                    "🚀 بدء الترميم",
                                    variant="primary",
                                    size="lg",
                                    elem_id="process_btn"
                                )
                            
                            # معلومات النظام
                            with gr.Column(scale=1, elem_classes="model-info"):
                                gr.Markdown("### 📊 معلومات النظام")
                                device_info = "GPU متاح" if torch.cuda.is_available() else "CPU فقط"
                                gr.Markdown(f"**الجهاز:** {device_info}")
                                gr.Markdown(f"**الإصدار:** {Config.VERSION}")
                                gr.Markdown(f"**المخرجات:** {Config.OUTPUT_DIR}")
                        
                        with gr.Column(scale=2):
                            # منطقة الصور
                            with gr.Row():
                                input_image = gr.Image(
                                    label="📤 الصورة المدخلة",
                                    type="numpy",
                                    height=400,
                                    elem_classes="image-container"
                                )
                                
                                output_image = gr.Image(
                                    label="📥 الصورة الناتجة",
                                    type="numpy",
                                    height=400,
                                    elem_classes="image-container"
                                )
                            
                            # منطقة الإحصائيات
                            stats_output = gr.JSON(
                                label="📈 إحصائيات المعالجة",
                                elem_classes="stats-box"
                            )
                    
                    # ربط الأحداث
                    process_btn.click(
                        fn=process_single_image,
                        inputs=[input_image, strength_slider, enhance_preprocess],
                        outputs=[output_image, stats_output]
                    )
                
                # علامة التبويب: المعالجة الدفعية
                with gr.TabItem("📁 معالجة دفعية", id="batch"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Column(scale=1, elem_classes="control-panel"):
                                gr.Markdown("### 📂 إعدادات الدفعة")
                                
                                folder_input = gr.File(
                                    label="اختر المجلد",
                                    file_count="directory",
                                    file_types=["image"]
                                )
                                
                                batch_strength = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="قوة التحسين للدفعة"
                                )
                                
                                batch_process_btn = gr.Button(
                                    "🚀 معالجة الدفعة",
                                    variant="primary",
                                    size="lg"
                                )
                        
                        with gr.Column(scale=2):
                            # معاينة النتائج
                            gr.Markdown("### 👁️ معاينة النتائج (أول 4 صور)")
                            with gr.Row():
                                batch_outputs = []
                                for i in range(4):
                                    with gr.Column(scale=1):
                                        output = gr.Image(
                                            label=f"النتيجة {i+1}",
                                            type="numpy",
                                            height=200,
                                            visible=False
                                        )
                                        batch_outputs.append(output)
                            
                            # إحصائيات الدفعة
                            batch_stats = gr.JSON(
                                label="📊 إحصائيات الدفعة",
                                elem_classes="stats-box"
                            )
                    
                    # ربط الأحداث للمعالجة الدفعية
                    batch_process_btn.click(
                        fn=process_batch_images,
                        inputs=[folder_input, batch_strength],
                        outputs=[gr.Gallery(value=batch_outputs), batch_stats]
                    )
                
                # علامة التبويب: التعليمات
                with gr.TabItem("❓ التعليمات", id="help"):
                    with gr.Column():
                        gr.Markdown("""
                        ## 📖 دليل الاستخدام
                        
                        ### 🎨 المعالجة الفردية
                        1. اختر صورة عن طريق السحب والإفلات أو النقر على منطقة الرفع
                        2. اضبط قوة التحسين حسب الرغبة
                        3. انقر على زر "بدء الترميم"
                        4. انتظر ظهور النتيجة والإحصائيات
                        
                        ### 📁 المعالجة الدفعية
                        1. اختر مجلد يحتوي على الصور
                        2. اضبط إعدادات التحسين
                        3. انقر على زر "معالجة الدفعة"
                        4. ستظهر أول 4 صور معاينة مع إحصائيات الدفعة
                        
                        ### ⚙️ الإعدادات المتقدمة
                        - **قوة التحسين**: تتحكم في شدة التحسين (1.0 هي القيمة المثالية)
                        - **تحسين مسبق**: يحسن جودة الصورة قبل المعالجة
                        
                        ### 💾 حفظ النتائج
                        - يتم حفظ جميع النتائج تلقائياً في: `{output_dir}`
                        - يمكنك العثور على الصور المحفوظة بالاسم والتاريخ
                        
                        ### 🛠️ المتطلبات الفنية
                        - مساحة ذاكرة وصول عشوائي: 4GB كحد أدنى (8GB موصى به)
                        - مساحة تخزين: 2GB للنموذج والنتائج
                        - دعم GPU: اختياري لكنه يسرع المعالجة
                        
                        ### ❗ ملاحظات هامة
                        - الخوارزمية الأساسية لتحسين الوجه محفوظة كما هي
                        - يتم تحسين باقي الجوانب فقط
                        - يدعم معظم صيغ الصور الشائعة
                        - الحد الأقصى لحجم الصورة: 4000x4000 بكسل
                        """.format(output_dir=Config.OUTPUT_DIR))
            
            # تذييل الصفحة
            gr.HTML("""
                <footer>
                    <p>Ultimate Face Restorer Pro v{version} | تم التطوير باستخدام GFPGAN وOpenCV | جميع الحقوق محفوظة © {year}</p>
                    <p style="font-size: 0.9em; opacity: 0.7;">تنويه: الخوارزمية الأساسية لتحسين الوجه محفوظة تماماً كما هي</p>
                </footer>
            """.format(version=Config.VERSION, year=datetime.now().year))
            
            # تحميل النموذج عند التشغيل
            def initialize_on_load():
                try:
                    manager = ModelManager()
                    manager.initialize_enhancer()
                    return "✅ النظام جاهز للاستخدام"
                except Exception as e:
                    return f"❌ خطأ في التهيئة: {str(e)}"
            
            demo.load(
                fn=initialize_on_load,
                outputs=[gr.Textbox(visible=False)]
            )
            
            return demo

# 11. التشغيل الرئيسي
def main():
    """الدالة الرئيسية للتشغيل"""
    print("=" * 60)
    print("Ultimate Face Restorer Pro - النسخة الاحترافية")
    print(f"الإصدار: {Config.VERSION}")
    print("=" * 60)
    
    try:
        # تهيئة النموذج
        print("🔧 جارٍ تهيئة النظام...")
        manager = ModelManager()
        manager.initialize_enhancer()
        
        # إنشاء الواجهة
        print("🚀 جارٍ تحميل الواجهة...")
        interface = AdvancedInterface.create_interface()
        
        # إعداد خيارات التشغيل
        server_name = "0.0.0.0"  # الاستماع على جميع الواجهات
        server_port = 7860
        share = True  # إنشاء رابط مشاركة عام
        
        print(f"🌐 جارٍ تشغيل الخادم على http://{server_name}:{server_port}")
        print(f"📎 رابط المشاركة: سينشأ تلقائياً عند التشغيل")
        print("=" * 60)
        print("✅ النظام جاهز! افتح المتصفح للبدء.")
        
        # تشغيل الواجهة
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            favicon_path=None,
            quiet=False,
            show_error=True,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ خطأ: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()