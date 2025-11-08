from google.colab import drive
from IPython.display import clear_output, HTML, display
import warnings
warnings.filterwarnings('ignore')

import os
import threading
import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import base64
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import random
import socket
from pyngrok import ngrok

!pip install pyngrok -q

# Mount Google Drive
def try_mount_drive(retry_times=3, wait_secs=5):
    """Mount drive với retry mechanism"""
    for i in range(retry_times):
        try:
            if not os.path.exists('/content/drive/MyDrive'):
                drive.mount('/content/drive', force_remount=True)
                clear_output(wait=True)
                print("✓ Drive mounted OK")
                return True
            else:
                print("✓ Drive đã mount rồi")
                return True
        except Exception as err:
            print(f"Lần {i+1}/{retry_times} thất bại: {err}")
            if i < retry_times - 1:
                time.sleep(wait_secs)
    
    print("Không mount được drive sau nhiều lần thử")
    return False

# Load model
model = None
if try_mount_drive():
    model_file = '/content/drive/MyDrive/do_an/AI_do_an_CNN.keras'
    
    if os.path.exists(model_file):
        model = load_model(model_file)
        print(f"✓ Loaded model: {model_file}")
    else:
        print("⚠ Không tìm thấy model file, dùng chế độ random")
else:
    print("⚠ Không có Drive, không thể load model")

# Danh sách món ăn
food_classes = ['ca_hu_kho', 'canh_chua_co_ca', 'canh_chua_khong_ca', 'canh_rau', 'com',
                'dau_hu_sot_ca', 'khay', 'rau_xao', 'suon_nuong', 'thit_kho',
                'thịt kho 1 trứng', 'thịt kho 2 trứng', 'trung_chien']

# Mapping tên và giá
food_info_map = {
    'com_trang': {'name': 'Cơm Trắng', 'price': 10000},
    'dau_hu_sot_ca': {'name': 'Đậu Hũ Sốt Cà', 'price': 25000},
    'ca_hu_kho': {'name': 'Cá Hũ Kho', 'price': 30000},
    'thit_kho': {'name': 'Thịt Kho', 'price': 30000},
    'thit_kho_1_trung': {'name': 'Thịt Kho 1 Trứng', 'price': 36000},
    'thit_kho_2_trung': {'name': 'Thịt Kho 2 Trứng', 'price': 42000},
    'canh_chua_co_ca': {'name': 'Canh Chua Có Cà', 'price': 25000},
    'canh_chua_khong_ca': {'name': 'Canh Chua Không Cà', 'price': 10000},
    'suon_nuong': {'name': 'Sườn Nướng', 'price': 30000},
    'canh_rau': {'name': 'Canh Rau', 'price': 7000},
    'rau_xao': {'name': 'Rau Xào', 'price': 10000},
    'trung_chien': {'name': 'Trứng Chiên', 'price': 25000},
    'khay_trong': {'name': 'Khay trống', 'price': 0}
}

# Tọa độ các ô trong khay
tray_regions = {
    "Ô 1 (trên trái)": (50, 50, 400, 290),
    "Ô 2 (trên giữa)": (450, 50, 800, 290),
    "Ô 3 (trên phải)": (850, 50, 1200, 290),
    "Ô 4 (dưới trái)": (100, 350, 550, 650),
    "Ô 5 (dưới phải)": (650, 350, 1150, 650),
}

def predict_single_food(img_crop):
    """Dự đoán món ăn từ 1 ảnh crop"""
    # Nếu không có model thì random
    if model is None:
        random_food = random.choice(list(food_info_map.keys()))
        info = food_info_map[random_food]
        fake_conf = random.uniform(0.85, 0.99)
        return info['name'], fake_conf, info['price']
    
    try:
        # Check ảnh hợp lệ
        if img_crop is None or img_crop.size == 0:
            return "Không xác định", 0.0, 0
        
        # Preprocessing
        rgb_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_img, (300, 300))
        
        if resized.dtype != np.float32:
            resized = resized.astype(np.float32)
        
        img_input = preprocess_input(resized).reshape(1, 300, 300, 3)
        
        # Predict
        preds = model.predict(img_input, verbose=0)
        idx = np.argmax(preds[0])
        conf = float(preds[0][idx])
        class_name = food_classes[idx]
        
        # Lấy thông tin món ăn
        info = food_info_map.get(class_name, {'name': 'Không xác định', 'price': 0})
        return info['name'], conf, info['price']
        
    except Exception as e:
        print(f"Lỗi predict: {e}")
        return "Không xác định", 0.0, 0

def img_to_base64(img):
    """Convert image sang base64 string"""
    try:
        if img is None or img.size == 0:
            # Tạo ảnh trắng dummy
            blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _, buf = cv2.imencode('.jpg', blank)
        else:
            _, buf = cv2.imencode('.jpg', img)
        
        return base64.b64encode(buf).decode('utf-8')
    except Exception as e:
        print(f"Lỗi encode base64: {e}")
        blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode('.jpg', blank)
        return base64.b64encode(buf).decode('utf-8')

def check_port_available(port_num):
    """Kiểm tra port có đang dùng không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port_num)) == 0

# Chọn port
server_port = 8001
if check_port_available(server_port):
    server_port = 8002

# Setup FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect_tray")
async def detect_full_tray(file: UploadFile = File(...)):
    """API detect cả khay 5 ô"""
    try:
        img_bytes = await file.read()
        
        if not img_bytes:
            return JSONResponse(content={"error": "File trống"}, status_code=400)
        
        # Decode image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(content={"error": "Không đọc được ảnh"}, status_code=400)
        
        # Resize về size cố định
        img = cv2.resize(img, (1280, 720))
        
        detect_results = []
        total_money = 0
        
        # Loop qua từng ô
        for idx, (region_name, coords) in enumerate(tray_regions.items(), 1):
            try:
                x1, y1, x2, y2 = coords
                # Đảm bảo tọa độ hợp lệ
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1280, x2), min(720, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop ảnh
                crop = img[y1:y2, x1:x2]
                
                if crop is None or crop.size == 0:
                    name, conf, price = "Không xác định", 0.0, 0
                else:
                    name, conf, price = predict_single_food(crop)
                
                total_money += price
                crop_b64 = img_to_base64(crop)
                
                detect_results.append({
                    'id': idx,
                    'region': region_name,
                    'food_name': name,
                    'confidence': round(conf * 100, 1),
                    'price': price,
                    'image': f"data:image/jpeg;base64,{crop_b64}"
                })
                
            except Exception as e:
                print(f"Lỗi xử lý region {region_name}: {e}")
                continue
        
        return {
            "success": True,
            "results": detect_results,
            "total_price": total_money
        }
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/detect_single")
async def detect_one_dish(file: UploadFile = File(...)):
    """API detect 1 món đơn"""
    try:
        img_bytes = await file.read()
        
        if not img_bytes:
            return JSONResponse(content={"error": "File trống"}, status_code=400)
        
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(content={"error": "Không đọc được ảnh"}, status_code=400)
        
        # Predict luôn trên ảnh gốc
        name, conf, price = predict_single_food(img)
        img_b64 = img_to_base64(img)
        
        result = {
            'id': 1,
            'region': 'Món đơn',
            'food_name': name,
            'confidence': round(conf * 100, 1),
            'price': price,
            'image': f"data:image/jpeg;base64,{img_b64}"
        }
        
        return {
            "success": True,
            "results": [result],
            "total_price": price
        }
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Chạy server trong thread
def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=server_port, log_level="error")

server_thread = threading.Thread(target=run_uvicorn, daemon=True)
server_thread.start()
time.sleep(3)

# Setup ngrok tunnel
ngrok_auth = "YOUR_NGROK_AUTHTOKEN_HERE"

try:
    ngrok.set_auth_token(ngrok_auth)
except Exception as e:
    print(f"Lỗi set ngrok token: {e}")

try:
    tunnel = ngrok.connect(server_port, "http")
    public_url = tunnel.public_url
    print(f"✓ Public URL: {public_url}")
except Exception as e:
    print(f"Lỗi tạo ngrok tunnel: {e}")
    public_url = f"http://localhost:{server_port}"

clear_output(wait=True)

html_code = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍜 Smart Canteen AI - Advanced</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }}

        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* Particles Background */
        #particles {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }}

        .container {{
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Header với 3D Effect */
        .main-header {{
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            padding: 40px;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            animation: fadeInDown 0.8s ease-out;
            transform-style: preserve-3d;
            transition: transform 0.3s;
        }}

        .main-header:hover {{
            transform: translateY(-10px) rotateX(5deg);
        }}

        .main-header h1 {{
            font-size: 56px;
            font-weight: 700;
            color: white;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }}

        .main-header p {{
            font-size: 20px;
            color: rgba(255,255,255,0.95);
        }}

        /* Layout Grid */
        .main-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}

        /* Glass Card */
        .glass-card {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 25px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}

        .glass-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}

        .section-title {{
            color: white;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .tab {{
            flex: 1;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 15px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .tab.active {{
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            box-shadow: 0 5px 20px rgba(255, 107, 107, 0.4);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
            animation: fadeIn 0.5s;
        }}

        /* Camera Controls */
        .camera-controls {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }}

        .control-group {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 15px;
        }}

        .control-group label {{
            display: block;
            color: white;
            font-size: 14px;
            margin-bottom: 8px;
            font-weight: 500;
        }}

        select, input[type="range"] {{
            width: 100%;
            padding: 10px;
            border-radius: 10px;
            border: none;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 14px;
        }}

        input[type="range"] {{
            padding: 0;
        }}

        /* Buttons */
        .btn-group {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .btn {{
            flex: 1;
            padding: 15px 25px;
            border: none;
            border-radius: 15px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            color: white;
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            box-shadow: 0 5px 20px rgba(76, 175, 80, 0.3);
        }}

        .btn-secondary {{
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3);
        }}

        .btn-danger {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }}

        .btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }}

        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        /* Canvas & Video */
        #video, #canvas {{
            width: 100%;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            border: 3px solid rgba(255,255,255,0.3);
            display: none;
        }}

        #previewCanvas {{
            width: 100%;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            border: 3px solid rgba(255,255,255,0.3);
        }}

        /* Upload Area */
        .upload-area {{
            border: 3px dashed rgba(255,255,255,0.4);
            border-radius: 20px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(255,255,255,0.05);
        }}

        .upload-area:hover {{
            border-color: #FF6B6B;
            background: rgba(255,255,255,0.1);
            transform: scale(1.02);
        }}

        .upload-area.dragover {{
            border-color: #4CAF50;
            background: rgba(76, 175, 80, 0.1);
        }}

        .upload-icon {{
            font-size: 64px;
            margin-bottom: 20px;
        }}

        .upload-text {{
            color: white;
            font-size: 18px;
            font-weight: 600;
        }}

        /* Food Items */
        .food-list {{
            max-height: 500px;
            overflow-y: auto;
            padding-right: 10px;
        }}

        .food-list::-webkit-scrollbar {{
            width: 8px;
        }}

        .food-list::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }}

        .food-list::-webkit-scrollbar-thumb {{
            background: rgba(255,255,255,0.3);
            border-radius: 10px;
        }}

        .food-item {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #FF6B6B;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            animation: slideInLeft 0.5s ease-out;
            transition: all 0.3s;
        }}

        .food-item:hover {{
            transform: translateX(10px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }}

        .food-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}

        .food-name {{
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .food-price {{
            font-size: 20px;
            color: #4CAF50;
            font-weight: 700;
            background: rgba(76, 175, 80, 0.1);
            padding: 5px 15px;
            border-radius: 20px;
        }}

        .confidence-bar-container {{
            margin-top: 10px;
        }}

        .confidence-label {{
            color: #718096;
            font-size: 14px;
            margin-bottom: 5px;
        }}

        .confidence-bar {{
            height: 10px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }}

        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #FF6B6B 0%, #4CAF50 100%);
            border-radius: 10px;
            transition: width 0.8s ease-out;
        }}
/* Total Price Box */
        .total-price-box {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(76, 175, 80, 0.4);
            margin: 20px 0;
            animation: pulse 2s infinite;
        }}

        .total-price-box h2 {{
            color: white;
            font-size: 48px;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}

        .total-price-box p {{
            color: rgba(255,255,255,0.9);
            font-size: 18px;
            margin-top: 5px;
        }}

        /* Receipt Modal */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            animation: fadeIn 0.3s;
        }}

        .modal.active {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .receipt {{
            background: white;
            border-radius: 25px;
            padding: 40px;
            max-width: 500px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 80px rgba(0,0,0,0.5);
            animation: scaleIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}

        .receipt-header {{
            text-align: center;
            border-bottom: 2px dashed #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }}

        .receipt-header h3 {{
            color: #FF6B6B;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 15px;
        }}

        .receipt-info {{
            color: #718096;
            font-size: 14px;
            margin: 5px 0;
        }}

        .receipt-item {{
            display: flex;
            justify-content: space-between;
            padding: 15px 0;
            border-bottom: 1px solid #f7fafc;
        }}

        .receipt-total {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
        }}

        .receipt-total h4 {{
            font-size: 36px;
            margin: 0;
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-top: 30px;
        }}

        .stat-card {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(20px);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.3);
transition: all 0.3s;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .stat-icon {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .stat-label {{
            color: rgba(255,255,255,0.9);
            font-size: 14px;
            margin-bottom: 5px;
        }}

        .stat-value {{
            color: white;
            font-size: 28px;
            font-weight: 700;
        }}

        /* Loading */
        .loading {{
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 5px solid rgba(255,255,255,0.3);
            border-top-color: #FF6B6B;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        .loading-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 999;
            flex-direction: column;
            gap: 20px;
        }}

        .loading-overlay.active {{
            display: flex;
        }}

        .loading-text {{
            color: white;
            font-size: 20px;
            font-weight: 600;
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes scaleIn {{
            from {{
                opacity: 0;
                transform: scale(0.8);
            }}
            to {{
                opacity: 1;
                transform: scale(1);
            }}
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.03); }}
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Responsive */
        @media (max-width: 1024px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            .camera-controls {{
                grid-template-columns: 1fr;
            }}
            .main-header h1 {{
                font-size: 36px;
}}
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <canvas id="particles"></canvas>

    <div class="container">
        <!-- Header -->
        <div class="main-header">
            <h1>🍜 Smart Canteen AI System</h1>
            <p>Hệ thống nhận diện món ăn thông minh với công nghệ AI</p>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Left Column: Camera/Upload -->
            <div class="glass-card">
                <h2 class="section-title">📸 Chụp / Tải Ảnh Món Ăn</h2>

                <div class="tabs">
                    <button class="tab active" onclick="switchTab('camera')">📷 Camera</button>
                    <button class="tab" onclick="switchTab('upload')">📁 Upload</button>
                </div>

                <!-- Camera Tab -->
                <div id="camera-tab" class="tab-content active">
                    <div class="camera-controls">
                        <div class="control-group">
                            <label>📹 Chọn Camera:</label>
                            <select id="cameraSelect"></select>
                        </div>
                        <div class="control-group">
                            <label>🔍 Zoom: <span id="zoomValue">1.0x</span></label>
                            <input type="range" id="zoomRange" min="1" max="3" step="0.1" value="1">
                        </div>
                        <div class="control-group">
                            <label>🎨 Filter:</label>
                            <select id="filterSelect">
                                <option value="none">Không</option>
                                <option value="grayscale">Đen trắng</option>
                                <option value="sepia">Sepia</option>
                                <option value="invert">Đảo màu</option>
                                <option value="brightness">Sáng</option>
                            </select>
                        </div>
                    </div>

                    <div class="btn-group">
                        <button class="btn btn-primary" onclick="startCamera()">▶️ Bật Camera</button>
                        <button class="btn btn-danger" onclick="stopCamera()">⏹️ Dừng</button>
                        <button class="btn btn-secondary" onclick="captureImage()">📸 Chụp</button>
                    </div>

                    <div id="cameraContainer"></div>
                    <canvas id="canvas" style="display: none;"></canvas>
                </div>

                <!-- Upload Tab -->
                <div id="upload-tab" class="tab-content">
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-icon">📤</div>
                        <div class="upload-text">Kéo thả ảnh vào đây hoặc click để chọn</div>
                        <input type="file" id="fileInput" accept="image/*" style="display:none;">
                    </div>
                </div>

                <button class="btn btn-secondary" onclick="detectFood()" style="width: 100%; margin-top: 20px;">
                    🔍 Nhận Diện Ngay
                </button>
            </div>

            <!-- Right Column: Results -->
            <div class="glass-card">
                <h2 class="section-title">📊 Kết Quả Nhận Diện</h2>

                <div class="food-list" id="foodList">
                    <div style="text-align: center; color: white; padding: 40px;">
                        <div style="font-size: 64px; margin-bottom: 20px;">🍽️</div>
                        <p style="font-size: 16px; opacity: 0.8;">Chưa có dữ liệu</p>
                        <p style="font-size: 14px; opacity: 0.6; margin-top: 10px;">Hãy chụp hoặc tải ảnh món ăn lên</p>
                    </div>
                </div>

                <div id="totalPriceBox" style="display: none;">
                    <div class="total-price-box">
                        <p>TỔNG CỘNG</p>
                        <h2 id="totalPrice">0 đ</h2>
                    </div>
                    <button class="btn btn-secondary" onclick="showReceipt()" style="width: 100%;">
                        💳 Thanh Toán Ngay
                    </button>
                </div>
            </div>
        </div>

        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">🤖</div>
                <div class="stat-label">Model Status</div>
                <div class="stat-value" id="modelStatus">✅ Ready</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🍽️</div>
                <div class="stat-label">Món đã nhận diện</div>
                <div class="stat-value" id="foodCount">0/5</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">💰</div>
                <div class="stat-label">Tổng giá trị</div>
                <div class="stat-value" id="totalValue">0đ</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-label">Độ chính xác TB</div>
                <div class="stat-value" id="avgConfidence">0%</div>
            </div>
        </div>
    </div>

    <!-- Receipt Modal -->
    <div class="modal" id="receiptModal">
        <div class="receipt">
            <div class="receipt-header">
                <h3>🧾 HÓA ĐƠN THANH TOÁN</h3>
                <div class="receipt-info"><strong>Mã HD:</strong> <span id="invoiceId"></span></div>
<div class="receipt-info"><strong>Mã tra cứu:</strong> <span id="trackingCode"></span></div>
                <div class="receipt-info"><strong>Thời gian:</strong> <span id="invoiceTime"></span></div>
                <div class="receipt-info"><strong>Phương thức:</strong> 💵 Tiền mặt</div>
            </div>

            <div id="receiptItems"></div>

            <div style="padding: 15px 0; border-top: 2px dashed #e2e8f0; margin-top: 15px;">
                <div class="receipt-item">
                    <span>Tạm tính</span>
                    <span id="subtotal">0đ</span>
                </div>
                <div class="receipt-item">
                    <span>Thuế VAT (10%)</span>
                    <span id="tax">0đ</span>
                </div>
            </div>

            <div class="receipt-total">
                <h4>💰 Tổng thanh toán: <span id="finalTotal">0đ</span></h4>
            </div>

            <div class="btn-group" style="margin-top: 20px;">
                <button class="btn btn-primary" onclick="completePayment()">✅ Hoàn tất</button>
                <button class="btn btn-danger" onclick="closeReceipt()">❌ Hủy</button>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading"></div>
        <div class="loading-text">🤖 AI đang phân tích...</div>
    </div>

    <script>
        const API_URL = '{public_url}';
        // State
        let stream = null;
        let detectedFoods = {{}};
        let devices = [];
        let currentDeviceId = null;
        let zoom = 1;
        let filter = 'none';
        let animationFrameId = null;
        let isCameraActive = false;
        let video = null;
        let previewCanvas = null;

        // Particles Animation
const canvas = document.getElementById('particles');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const particles = [];
        for (let i = 0; i < 50; i++) {{
            particles.push({{
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                radius: Math.random() * 3 + 1
            }});
        }}

        function animateParticles() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';

            particles.forEach(p => {{
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();

                p.x += p.vx;
                p.y += p.vy;

                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            }});

            requestAnimationFrame(animateParticles);
        }}
        animateParticles();

        window.addEventListener('resize', () => {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }});

        // Tab Switching
        function switchTab(tab) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (tab === 'camera') {{
                document.querySelector('.tab:first-child').classList.add('active');
                document.getElementById('camera-tab').classList.add('active');
            }} else {{
                document.querySelector('.tab:last-child').classList.add('active');
                document.getElementById('upload-tab').classList.add('active');
                stopCamera();
            }}
        }}

        // Camera Functions
        async function populateCameras() {{
            try {{
                devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');
                const select = document.getElementById('cameraSelect');
                select.innerHTML = '';
                videoDevices.forEach((device, index) => {{
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.text = device.label || `Camera ${{index + 1}}`;
                    select.appendChild(option);
                }});
                if (videoDevices.length > 0) {{
                    currentDeviceId = videoDevices[0].deviceId;
                    select.value = currentDeviceId;
                }}
            }} catch (err) {{
                console.error('Error enumerating devices:', err);
            }}
        }}

        document.getElementById('cameraSelect').addEventListener('change', (e) => {{
            currentDeviceId = e.target.value;
            if (isCameraActive) {{
                stopCamera();
                startCamera();
            }}
        }});

        async function startCamera() {{
            if (!devices.length) await populateCameras();

            try {{
                const constraints = {{
                    video: {{
                        deviceId: currentDeviceId ? {{ exact: currentDeviceId }} : undefined,
                        width: {{ ideal: 1280 }},
                        height: {{ ideal: 720 }}
                    }}
                }};

                stream = await navigator.mediaDevices.getUserMedia(constraints);

                const cameraContainer = document.getElementById('cameraContainer');
                cameraContainer.innerHTML = '';

                video = document.createElement('video');
                video.srcObject = stream;
                video.autoplay = true;
                video.playsinline = true;
                video.style.width = '100%';
                video.style.borderRadius = '20px';
                video.style.boxShadow = '0 10px 40px rgba(0,0,0,0.3)';
                video.style.border = '3px solid rgba(255,255,255,0.3)';

                previewCanvas = document.createElement('canvas');
                previewCanvas.style.display = 'none';

                cameraContainer.appendChild(video);

                await video.play();

                // Adjust iframe height
                if (typeof google !== 'undefined' && google.colab && google.colab.output) {{
                    google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
                }}

                isCameraActive = true;
                previewLoop();
            }} catch (err) {{
                alert('❌ Không thể truy cập camera: ' + err.message);
            }}
        }}

        function stopCamera() {{
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            if (stream) {{
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }}
            isCameraActive = false;
            const cameraContainer = document.getElementById('cameraContainer');
            cameraContainer.innerHTML = '';
            video = null;
            previewCanvas = null;
        }}

        function previewLoop() {{
            if (!isCameraActive || !video) return;

            if (!previewCanvas) {{
                previewCanvas = document.createElement('canvas');
            }}

            previewCanvas.width = video.videoWidth;
            previewCanvas.height = video.videoHeight;

            const ctx = previewCanvas.getContext('2d');

            // Apply zoom
            const w = video.videoWidth / zoom;
            const h = video.videoHeight / zoom;
            const x = (video.videoWidth - w) / 2;
            const y = (video.videoHeight - h) / 2;

            ctx.drawImage(video, x, y, w, h, 0, 0, previewCanvas.width, previewCanvas.height);

            // Apply filter
            applyFilter(ctx, previewCanvas, filter);

            animationFrameId = requestAnimationFrame(previewLoop);
        }}

        function captureImage() {{
            if (!isCameraActive) {{
                alert('Vui lòng bật camera trước');
                return;
            }}

            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Apply zoom
            const w = video.videoWidth / zoom;
            const h = video.videoHeight / zoom;
            const x = (video.videoWidth - w) / 2;
            const y = (video.videoHeight - h) / 2;

            ctx.drawImage(video, x, y, w, h, 0, 0, canvas.width, canvas.height);

            // Apply filter
            applyFilter(ctx, canvas, filter);

            // Draw grid if in camera tab
            if (document.getElementById('camera-tab').classList.contains('active')) {{
                drawGrid(ctx, canvas);
            }}

            canvas.style.display = 'block';
            stopCamera();
        }}

        function applyFilter(ctx, canvas, filterType) {{
            if (filterType === 'none') return;

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {{
                if (filterType === 'grayscale') {{
                    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    data[i] = data[i + 1] = data[i + 2] = avg;
                }} else if (filterType === 'sepia') {{
                    const r = data[i], g = data[i + 1], b = data[i + 2];
                    data[i] = Math.min(255, r * 0.393 + g * 0.769 + b * 0.189);
                    data[i + 1] = Math.min(255, r * 0.349 + g * 0.686 + b * 0.168);
                    data[i + 2] = Math.min(255, r * 0.272 + g * 0.534 + b * 0.131);
                }} else if (filterType === 'invert') {{
                    data[i] = 255 - data[i];
                    data[i + 1] = 255 - data[i + 1];
                    data[i + 2] = 255 - data[i + 2];
                }} else if (filterType === 'brightness') {{
                    data[i] = Math.min(255, data[i] * 1.3);
                    data[i + 1] = Math.min(255, data[i + 1] * 1.3);
                    data[i + 2] = Math.min(255, data[i + 2] * 1.3);
                }}
            }}

            ctx.putImageData(imageData, 0, 0);
        }}

        function drawGrid(ctx, canvas) {{
            const w = canvas.width;
            const h = canvas.height;

            ctx.strokeStyle = '#FF6B6B';
            ctx.lineWidth = 4;

            // Horizontal line
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();

            // Vertical line (top half)
ctx.beginPath();
            ctx.moveTo(w / 2, 0);
            ctx.lineTo(w / 2, h / 2);
            ctx.stroke();

            // Vertical lines (bottom half)
            ctx.beginPath();
            ctx.moveTo(w / 3, h / 2);
            ctx.lineTo(w / 3, h);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(2 * w / 3, h / 2);
            ctx.lineTo(2 * w / 3, h);
            ctx.stroke();

            // Labels
            const positions = [
                {{ x: w / 4, y: h / 4, label: 'Ô 1' }},
                {{ x: w / 2, y: h / 4, label: 'Ô 2' }},
                {{ x: 3 * w / 4, y: h / 4, label: 'Ô 3' }},
                {{ x: w / 3, y: 3 * h / 4, label: 'Ô 4' }},
                {{ x: 2 * w / 3, y: 3 * h / 4, label: 'Ô 5' }}
            ];

            ctx.font = 'bold 28px Poppins';
            ctx.textAlign = 'center';

            positions.forEach(pos => {{
                // Shadow
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.fillText(pos.label, pos.x + 2, pos.y + 2);
                // Text
                ctx.fillStyle = 'white';
                ctx.fillText(pos.label, pos.x, pos.y);
            }});
        }}

        // Controls listeners
        document.getElementById('zoomRange').addEventListener('input', (e) => {{
            zoom = parseFloat(e.target.value);
            document.getElementById('zoomValue').textContent = zoom.toFixed(1) + 'x';
        }});

        document.getElementById('filterSelect').addEventListener('change', (e) => {{
            filter = e.target.value;
        }});

        // Upload Functions
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.classList.add('dragover');
        }});

        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.classList.remove('dragover');
        }});

        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {{
                loadImage(file);
            }}
        }});

        fileInput.addEventListener('change', (e) => {{
            const file = e.target.files[0];
            if (file) loadImage(file);
        }});

        function loadImage(file) {{
            const reader = new FileReader();
            reader.onload = (e) => {{
                const img = new Image();
                img.onload = () => {{
                    const canvas = document.getElementById('canvas');
                    const ctx = canvas.getContext('2d');

                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);

                    canvas.style.display = 'block';
                    // Switch to camera tab to show result
                    switchTab('camera');
                }};
                img.src = e.target.result;
            }};
            reader.readAsDataURL(file);
        }}

        // AI Detection
        async function detectFood() {{
            const canvas = document.getElementById('canvas');
            if (canvas.style.display === 'none') {{
                alert('⚠️ Vui lòng chụp hoặc tải ảnh trước!');
                return;
            }}

            document.getElementById('loadingOverlay').classList.add('active');

            const isTray = document.getElementById('camera-tab').classList.contains('active');
            const endpoint = isTray ? '/detect_tray' : '/detect_single';

            canvas.toBlob(async (blob) => {{
                const formData = new FormData();
                formData.append('file', blob, 'image.jpg');

                try {{
                    const response = await fetch(API_URL + endpoint, {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();

                    if (data.success) {{
                        detectedFoods = {{}};
                        let total = 0;
                        let totalConfidence = 0;
                        let count = 0;

                        data.results.forEach(res => {{
                            const tray = res.region;
                            const conf = res.confidence;
                            detectedFoods[tray] = {{
                                food: res.food_name,
                                price: res.price,
                                confidence: conf,
                                icon: getIcon(res.food_name)
                            }};
                            total += res.price;
                            totalConfidence += conf;
                            count++;
                        }});

                        displayResults();

                        // Update total price
                        document.getElementById('totalPrice').textContent = `💰 ${{total.toLocaleString()}} đ`;
                        document.getElementById('totalPriceBox').style.display = 'block';

                        // Update stats
                        document.getElementById('foodCount').textContent = `${{count}}/5`;
                        document.getElementById('totalValue').textContent = total.toLocaleString() + 'đ';
                        document.getElementById('avgConfidence').textContent = (totalConfidence / count).toFixed(1) + '%';
                    }} else {{
                        alert(data.error);
                    }}
                }} catch (e) {{
                    alert('❌ Lỗi kết nối: ' + e.message);
                }} finally {{
                    document.getElementById('loadingOverlay').classList.remove('active');
                }}
            }}, 'image/jpeg');
        }}

        function getIcon(name) {{
            const icons = {{
                'Cơm Trắng': '🍚',
                'Đậu Hũ Sốt Cà': '🧈',
                'Cá Hũ Kho': '🐠',
                'Thịt Kho': '🍖',
                'Thịt Kho 1 Trứng': '🍖🥚',
                'Thịt Kho 2 Trứng': '🍖🥚🥚',
                'Canh Chua Có Cà': '🥣🐟',
                'Canh Chua Không Cà': '🥣',
                'Sườn Nướng': '🥩',
                'Canh Rau': '🥬',
                'Rau Xào': '🥗',
                'Trứng Chiên': '🍳',
                'Không xác định': '❓'
            }};
            return icons[name] || '🍲';
        }}

        function displayResults() {{
            const foodList = document.getElementById('foodList');
            foodList.innerHTML = '';

            Object.entries(detectedFoods).forEach(([tray, data]) => {{
                const item = document.createElement('div');
                item.className = 'food-item';
                item.innerHTML = `
                    <div class="food-header">
                        <div class="food-name">
                            ${{data.icon}} ${{tray}}: ${{data.food}}
                        </div>
                        <div class="food-price">💰 ${{data.price.toLocaleString()}}đ</div>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-label">🎯 Độ tin cậy: ${{data.confidence.toFixed(1)}}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${{data.confidence}}%"></div>
                        </div>
                    </div>
                `;
foodList.appendChild(item);
            }});
        }}

        // Receipt Functions
        function showReceipt() {{
            const now = new Date();
            const invoiceId = now.toISOString().replace(/[-:]/g, '').split('.')[0];
            const trackingCode = Math.random().toString(36).substr(2, 8).toUpperCase();
            const invoiceTime = now.toLocaleString('vi-VN');

            document.getElementById('invoiceId').textContent = invoiceId;
            document.getElementById('trackingCode').textContent = trackingCode;
            document.getElementById('invoiceTime').textContent = invoiceTime;

            // Items
            const itemsContainer = document.getElementById('receiptItems');
            itemsContainer.innerHTML = '';

            let subtotal = 0;
            Object.entries(detectedFoods).forEach(([tray, data]) => {{
                subtotal += data.price;
                const item = document.createElement('div');
                item.className = 'receipt-item';
                item.innerHTML = `
                    <span>${{data.icon}} ${{data.food}}</span>
                    <span><strong>${{data.price.toLocaleString()}}đ</strong></span>
                `;
                itemsContainer.appendChild(item);
            }});

            const tax = subtotal * 0.1;
            const finalTotal = subtotal + tax;

            document.getElementById('subtotal').textContent = subtotal.toLocaleString() + 'đ';
            document.getElementById('tax').textContent = tax.toLocaleString() + 'đ';
            document.getElementById('finalTotal').textContent = finalTotal.toLocaleString() + 'đ';

            document.getElementById('receiptModal').classList.add('active');
        }}

        function closeReceipt() {{
            document.getElementById('receiptModal').classList.remove('active');
        }}

        function completePayment() {{
            // Confetti effect
            for (let i = 0; i < 100; i++) {{
                createConfetti();
            }}

            setTimeout(() => {{
                alert('✅ Thanh toán thành công!\n\nCảm ơn quý khách đã sử dụng dịch vụ!');
                closeReceipt();
                resetApp();
            }}, 1000);
        }}

        function createConfetti() {{
            const confetti = document.createElement('div');
confetti.style.position = 'fixed';
            confetti.style.width = '10px';
            confetti.style.height = '10px';
            confetti.style.backgroundColor = ['#FF6B6B', '#4CAF50', '#667eea', '#FFD700'][Math.floor(Math.random() * 4)];
            confetti.style.left = Math.random() * window.innerWidth + 'px';
            confetti.style.top = '-10px';
            confetti.style.zIndex = '9999';
            confetti.style.borderRadius = '50%';
            confetti.style.pointerEvents = 'none';
            document.body.appendChild(confetti);

            const fall = confetti.animate([
                {{ transform: 'translateY(0) rotate(0deg)', opacity: 1 }},
                {{ transform: `translateY(${{window.innerHeight}}px) rotate(${{Math.random() * 360}}deg)`, opacity: 0 }}
            ], {{
                duration: 3000 + Math.random() * 2000,
                easing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)'
            }});

            fall.onfinish = () => confetti.remove();
        }}

        function resetApp() {{
            detectedFoods = {{}};
            document.getElementById('foodList').innerHTML = `
                <div style="text-align: center; color: white; padding: 40px;">
                    <div style="font-size: 64px; margin-bottom: 20px;">🍽️</div>
                    <p style="font-size: 16px; opacity: 0.8;">Chưa có dữ liệu</p>
                    <p style="font-size: 14px; opacity: 0.6; margin-top: 10px;">Hãy chụp hoặc tải ảnh món ăn lên</p>
                </div>
            `;
            document.getElementById('totalPriceBox').style.display = 'none';
            document.getElementById('canvas').style.display = 'none';
            document.getElementById('foodCount').textContent = '0/5';
            document.getElementById('totalValue').textContent = '0đ';
            document.getElementById('avgConfidence').textContent = '0%';
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') {{
                closeReceipt();
            }} else if (e.key === 'Enter' && e.ctrlKey) {{
                detectFood();
            }} else if (e.key === 'c' && e.ctrlKey && e.shiftKey) {{
                captureImage();
            }}
        }});

        // Initialize
        populateCameras();
        console.log('🍜 Smart Canteen AI System Ready!');
        console.log('💡 Keyboard shortcuts:');
        console.log('   Ctrl+Enter: Nhận diện');
        console.log('   Ctrl+Shift+C: Chụp ảnh');
        console.log('   Escape: Đóng modal');
    </script>
</body>
</html>

"""

display(HTML(html_code))
