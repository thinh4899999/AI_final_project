import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from datetime import datetime
import json

FOOD_ITEMS = {
    'trung_chien': {'name': 'Trứng chiên', 'price': 10000},
    'thit_kho_2trung': {'name': 'Thịt kho 2 trứng', 'price': 25000},
    'thit_kho_1trung': {'name': 'Thịt kho 1 trứng', 'price': 18000},
    'thit_kho': {'name': 'Thịt kho', 'price': 20000},
    'suon_nuong': {'name': 'Sườn nướng', 'price': 20000},
    'rau_luoc': {'name': 'Rau luộc', 'price': 10000},
    'kim_chi': {'name': 'Kim chi', 'price': 15000},
    'khay_trong': {'name': 'Khay trống', 'price': 0},
    'dau_que': {'name': 'Đậu que', 'price': 10000},
    'dau_hu': {'name': 'Đậu hũ', 'price': 5000},
    'com_trang': {'name': 'Cơm trắng', 'price': 10000},
    'canh_rau': {'name': 'Canh rau', 'price': 15000},
    'canh_chua_co_ca': {'name': 'Canh chua có cá', 'price': 35000},
    'canh_chua': {'name': 'Canh chua', 'price': 35000},
    'ca_hu_kho': {'name': 'Cá hú kho', 'price': 30000}
}

PAYMENT_METHODS = {
    'cash': '💵 Tiền mặt',
    'card': '💳 Thẻ ngân hàng',
    'momo': '📱 MoMo',
    'zalopay': '💙 ZaloPay',
    'banking': '🏦 Chuyển khoản'
}

# ===============================
# 🎨 PAGE CONFIGURATION
# ===============================

st.set_page_config(
    page_title="Canteen Food Recognition",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .food-box {
        border: 3px solid #4ECDC4;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .price-tag {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FFD93D;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .total-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .grid-overlay {
        border: 2px solid #FF6B6B;
        border-radius: 10px;
    }
    .payment-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ===============================
# 🔧 UTILITY FUNCTIONS
# ===============================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('ULTIMATE_FOOD_RECOGNITION_MODEL.keras')
        return model
    except Exception as e:
        st.error(f"❌ Không thể tải model: {e}")
        st.info("💡 Vui lòng đảm bảo file 'ULTIMATE_FOOD_RECOGNITION_MODEL.keras' có trong thư mục!")
        return None


def preprocess_image(img, target_size=(300, 300)):
    """Preprocess image for model prediction"""
    img = cv2.resize(img, target_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
    return img


def split_tray_image(image):
    """
    Split tray image into 5 sections:
    - Top row: 3 sections
    - Bottom row: 2 sections
    """
    h, w = image.shape[:2]

    # Calculate dimensions
    top_h = h // 2
    bottom_h = h - top_h
    top_w = w // 3
    bottom_w = w // 2

    sections = []

    # Top row - 3 sections
    for i in range(3):
        section = image[0:top_h, i * top_w:(i + 1) * top_w]
        sections.append(section)

    # Bottom row - 2 sections
    for i in range(2):
        section = image[top_h:h, i * bottom_w:(i + 1) * bottom_w]
        sections.append(section)

    return sections


def draw_grid_on_image(image):
    """Draw grid overlay on image"""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Vertical lines for top row
    cv2.line(overlay, (w // 3, 0), (w // 3, h // 2), (255, 107, 107), 3)
    cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h // 2), (255, 107, 107), 3)

    # Horizontal line
    cv2.line(overlay, (0, h // 2), (w, h // 2), (255, 107, 107), 3)

    # Vertical line for bottom row
    cv2.line(overlay, (w // 2, h // 2), (w // 2, h), (255, 107, 107), 3)

    return overlay


def predict_food(model, image):
    """Predict food from image"""
    processed = preprocess_image(image)
    processed = np.expand_dims(processed, axis=0)

    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]

    class_names = list(FOOD_ITEMS.keys())
    predicted_class = class_names[class_idx]

    return predicted_class, confidence


def format_currency(amount):
    """Format currency in Vietnamese Dong"""
    return f"{amount:,.0f}đ".replace(',', '.')


# ===============================
# 🎯 MAIN APPLICATION
# ===============================

def main():
    # Header
    st.markdown('<div class="main-header">🍽️ HỆ THỐNG NHẬN DIỆN MÓN ĂN CANTEEN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Đưa khay cơm vào camera để tự động nhận diện và tính tiền</div>',
                unsafe_allow_html=True)

    # Initialize session state
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    if 'payment_complete' not in st.session_state:
        st.session_state.payment_complete = False

    # Sidebar - Menu & Settings
    with st.sidebar:
        st.header("📋 MENU MÓN ĂN")
        st.markdown("---")

        for key, item in FOOD_ITEMS.items():
            if key != 'khay_trong':
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 10px; margin: 5px 0; border-radius: 10px; color: white;'>
                    <b>{item['name']}</b><br>
                    <span style='color: #FFD93D; font-size: 1.2rem;'>{format_currency(item['price'])}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.info(
            "💡 **Hướng dẫn sử dụng:**\n\n1. Chụp ảnh khay cơm hoặc tải ảnh lên\n2. Hệ thống tự động chia 5 khung và nhận diện\n3. Kiểm tra giỏ hàng và chọn phương thức thanh toán")

    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Nhận diện", "🛒 Giỏ hàng", "💳 Thanh toán"])

    # Tab 1: Recognition
    with tab1:
        st.header("📸 Chụp hoặc tải ảnh khay cơm")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Load model
            model = load_model()

            if model is None:
                st.stop()

            # Input method selection
            input_method = st.radio("Chọn phương thức nhập ảnh:",
                                    ["📤 Tải ảnh lên", "📷 Chụp từ camera"],
                                    horizontal=True)

            image = None

            if input_method == "📤 Tải ảnh lên":
                uploaded_file = st.file_uploader("Chọn ảnh khay cơm", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    image = np.array(image)
                    if len(image.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            else:  # Camera capture
                camera_image = st.camera_input("Chụp ảnh khay cơm")
                if camera_image:
                    image = Image.open(camera_image)
                    image = np.array(image)

            if image is not None:
                # Show original image with grid overlay
                st.subheader("🖼️ Ảnh gốc với lưới phân chia")
                grid_image = draw_grid_on_image(image.copy())
                st.image(grid_image, use_container_width=True)

                # Process button
                if st.button("🚀 NHẬN DIỆN MÓN ĂN", type="primary", use_container_width=True):
                    with st.spinner("🔍 Đang phân tích khay cơm..."):
                        # Split image into 5 sections
                        sections = split_tray_image(image)

                        # Display sections and predictions
                        st.subheader("📊 Kết quả nhận diện từng khu vực")

                        detected_items = []

                        # Row 1: 3 sections
                        cols = st.columns(3)
                        for i in range(3):
                            with cols[i]:
                                st.image(sections[i], caption=f"Khu vực {i + 1}", use_container_width=True)

                                # Predict
                                pred_class, confidence = predict_food(model, sections[i])
                                food_info = FOOD_ITEMS[pred_class]

                                if pred_class != 'khay_trong':
                                    st.markdown(f"""
                                    <div class='food-box'>
                                        <div style='font-size: 1.2rem;'><b>{food_info['name']}</b></div>
                                        <div>Độ tin cậy: {confidence * 100:.1f}%</div>
                                        <div class='price-tag'>{format_currency(food_info['price'])}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    detected_items.append(pred_class)
                                else:
                                    st.info("Khay trống")

                        # Row 2: 2 sections
                        cols = st.columns([1, 2, 1])
                        for i in range(2):
                            with cols[i]:
                                st.image(sections[i + 3], caption=f"Khu vực {i + 4}", use_container_width=True)

                                # Predict
                                pred_class, confidence = predict_food(model, sections[i + 3])
                                food_info = FOOD_ITEMS[pred_class]

                                if pred_class != 'khay_trong':
                                    st.markdown(f"""
                                    <div class='food-box'>
                                        <div style='font-size: 1.2rem;'><b>{food_info['name']}</b></div>
                                        <div>Độ tin cậy: {confidence * 100:.1f}%</div>
                                        <div class='price-tag'>{format_currency(food_info['price'])}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    detected_items.append(pred_class)
                                else:
                                    st.info("Khay trống")

                        # Add to cart
                        if detected_items:
                            if st.button("➕ THÊM VÀO GIỎ HÀNG", type="primary", use_container_width=True):
                                order = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'items': detected_items
                                }
                                st.session_state.cart.append(order)
                                st.success("✅ Đã thêm vào giỏ hàng!")
                                st.balloons()

        with col2:
            st.info("""
            ### 📐 Cách bố trí khay

            Khay cơm được chia thành 5 khu vực:

            ```
            ┌───┬───┬───┐
            │ 1 │ 2 │ 3 │  ← Hàng trên
            ├───┴───┴───┤
            │  4  │  5  │  ← Hàng dưới
            └─────┴─────┘
            ```

            **Mẹo:**
            - Đặt khay cơm thẳng
            - Ánh sáng đủ sáng
            - Camera không bị rung
            """)

    # Tab 2: Cart
    with tab2:
        st.header("🛒 Giỏ hàng")

        if not st.session_state.cart:
            st.info("🛒 Giỏ hàng trống. Hãy nhận diện món ăn để thêm vào giỏ!")
        else:
            for idx, order in enumerate(st.session_state.cart):
                with st.expander(f"📦 Đơn hàng #{idx + 1} - {order['timestamp']}", expanded=True):
                    total = 0
                    for item_key in order['items']:
                        item = FOOD_ITEMS[item_key]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{item['name']}**")
                        with col2:
                            st.write(f"{format_currency(item['price'])}")
                        total += item['price']

                    st.markdown(f"""
                    <div class='total-box' style='font-size: 1.5rem; padding: 10px;'>
                        Tổng: {format_currency(total)}
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"🗑️ Xóa đơn hàng #{idx + 1}", key=f"del_{idx}"):
                        st.session_state.cart.pop(idx)
                        st.rerun()

            # Total all orders
            grand_total = sum(
                sum(FOOD_ITEMS[item]['price'] for item in order['items'])
                for order in st.session_state.cart
            )

            st.markdown("---")
            st.markdown(f"""
            <div class='total-box'>
                TỔNG CỘNG: {format_currency(grand_total)}
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ XÓA TẤT CẢ", type="secondary", use_container_width=True):
                    st.session_state.cart = []
                    st.rerun()
            with col2:
                if st.button("💳 THANH TOÁN", type="primary", use_container_width=True):
                    st.switch_page

    # Tab 3: Payment
    with tab3:
        st.header("💳 Thanh toán")

        if not st.session_state.cart:
            st.warning("⚠️ Giỏ hàng trống! Vui lòng thêm món ăn trước khi thanh toán.")
        else:
            # Calculate total
            grand_total = sum(
                sum(FOOD_ITEMS[item]['price'] for item in order['items'])
                for order in st.session_state.cart
            )

            # Payment summary
            st.subheader("📋 Chi tiết đơn hàng")

            item_counts = {}
            for order in st.session_state.cart:
                for item_key in order['items']:
                    item_counts[item_key] = item_counts.get(item_key, 0) + 1

            for item_key, count in item_counts.items():
                item = FOOD_ITEMS[item_key]
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(f"**{item['name']}**")
                with col2:
                    st.write(f"x{count}")
                with col3:
                    st.write(f"{format_currency(item['price'] * count)}")

            st.markdown("---")
            st.markdown(f"""
            <div class='total-box'>
                TỔNG TIỀN: {format_currency(grand_total)}
            </div>
            """, unsafe_allow_html=True)

            # Payment method selection
            st.subheader("💳 Chọn phương thức thanh toán")

            payment_method = st.radio(
                "Phương thức thanh toán:",
                options=list(PAYMENT_METHODS.keys()),
                format_func=lambda x: PAYMENT_METHODS[x],
                horizontal=True
            )

            # Additional info based on payment method
            if payment_method == 'cash':
                st.info("💵 Vui lòng chuẩn bị tiền mặt và thanh toán tại quầy.")
            elif payment_method in ['momo', 'zalopay']:
                st.info(f"📱 Vui lòng quét mã QR để thanh toán qua {PAYMENT_METHODS[payment_method]}")
                # Placeholder for QR code
                st.image("https://via.placeholder.com/300x300?text=QR+Code", width=300)
            elif payment_method == 'card':
                st.info("💳 Vui lòng đưa thẻ vào máy POS để thanh toán.")
            elif payment_method == 'banking':
                st.info("""
                🏦 **Thông tin chuyển khoản:**
                - Ngân hàng: Vietcombank
                - Số tài khoản: 1234567890
                - Chủ tài khoản: CANTEEN SYSTEM
                - Nội dung: CANTEEN [Số điện thoại]
                """)

            # Confirm payment button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ XÁC NHẬN THANH TOÁN", type="primary", use_container_width=True):
                st.session_state.payment_complete = True

                # Create receipt
                receipt = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'items': item_counts,
                    'total': grand_total,
                    'payment_method': PAYMENT_METHODS[payment_method]
                }

                # Show success message
                st.markdown("""
                <div class='success-message'>
                    <h2>✅ THANH TOÁN THÀNH CÔNG!</h2>
                    <p>Cảm ơn bạn đã sử dụng dịch vụ!</p>
                </div>
                """, unsafe_allow_html=True)

                st.balloons()

                # Show receipt
                with st.expander("🧾 Xem hóa đơn", expanded=True):
                    st.write(f"**Thời gian:** {receipt['timestamp']}")
                    st.write(f"**Phương thức:** {receipt['payment_method']}")
                    st.markdown("---")

                    for item_key, count in receipt['items'].items():
                        item = FOOD_ITEMS[item_key]
                        st.write(f"{item['name']} x{count}: {format_currency(item['price'] * count)}")

                    st.markdown("---")
                    st.markdown(f"**TỔNG CỘNG: {format_currency(receipt['total'])}**")

                # Clear cart after 3 seconds
                if st.button("🔄 ĐƠN MỚI"):
                    st.session_state.cart = []
                    st.session_state.payment_complete = False
                    st.rerun()


if __name__ == "__main__":
    main()