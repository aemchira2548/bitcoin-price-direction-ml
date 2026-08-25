"""
Streamlit App: Bitcoin Price Direction Prediction
โปรเจค Machine Learning - ทำนายทิศทางราคา Bitcoin (ขึ้น/ลง วันถัดไป)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

st.set_page_config(
    page_title="Bitcoin Direction Predictor",
    page_icon="₿",
    layout="wide",
)

BASE = Path(__file__).parent

# ---------- Custom styling (teal + monospace accents) ----------
st.markdown("""
<style>
    :root {
        --teal-dark: #013A40;
        --teal: #028090;
        --seafoam: #00A896;
        --mint: #02C39A;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--teal-dark);
    }
    section[data-testid="stSidebar"] * {
        color: #F7FAFA !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-family: 'Courier New', monospace;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Headings */
    h1 {
        color: var(--teal-dark);
        font-weight: 800;
        border-bottom: 2px solid var(--mint);
        padding-bottom: 0.3rem;
    }
    h2, h3 {
        color: var(--teal-dark);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #F7FAFA;
        border: 1px solid #DCEBEA;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Courier New', monospace;
        color: var(--teal);
    }
    div[data-testid="stMetricValue"] {
        color: var(--teal-dark);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--teal);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: var(--seafoam);
        color: white;
    }

    /* Code / monospace blocks */
    code, .stCode {
        font-family: 'Courier New', monospace !important;
    }

    /* DataFrames */
    div[data-testid="stDataFrame"] {
        border: 1px solid #DCEBEA;
        border-radius: 8px;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 700;
        color: var(--teal-dark);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load data & models ----------
@st.cache_data
def load_data():
    raw = pd.read_csv(BASE / "data/bitcoin_clean.csv", parse_dates=["Date"])
    features = pd.read_csv(BASE / "data/bitcoin_features.csv", parse_dates=["Date"])
    comparison = pd.read_csv(BASE / "data/model_comparison.csv")
    return raw, features, comparison

@st.cache_resource
def load_models():
    with open(BASE / "models/best_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    with open(BASE / "models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE / "models/all_models.pkl", "rb") as f:
        all_models = pickle.load(f)
    with open(BASE / "models/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return best_model, scaler, all_models, meta

raw_df, feat_df, comparison_df = load_data()
best_model, scaler, all_models, meta = load_models()
FEATURES = meta["features"]

# ---------- Sidebar navigation ----------
st.sidebar.title("₿ Bitcoin ML Project")
page = st.sidebar.radio(
    "เมนู",
    ["ข้อมูลผู้พัฒนา", "1. โจทย์ & Dataset", "2. Data Preprocessing",
     "3. โมเดล ML", "4. เปรียบเทียบผลลัพธ์", "5. ทำนายผล (Live Demo)"],
)

# =====================================================================
# หน้า: ข้อมูลผู้พัฒนา
# =====================================================================
if page == "ข้อมูลผู้พัฒนา":
    st.title("ข้อมูลผู้พัฒนา")

    st.markdown("""
    <style>
        .dev-card {
            background: linear-gradient(135deg, #013A40 0%, #028090 100%);
            border-radius: 20px;
            padding: 2.2rem 2.5rem;
            display: flex;
            align-items: center;
            gap: 2.5rem;
            box-shadow: 0 8px 24px rgba(1,58,64,0.25);
            margin-bottom: 1.5rem;
        }
        .dev-photo-wrap {
            flex-shrink: 0;
        }
        .dev-photo-wrap img {
            width: 170px;
            height: 220px;
            object-fit: cover;
            border-radius: 16px;
            border: 3px solid #02C39A;
            box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        }
        .dev-info h2 {
            color: white;
            font-size: 1.7rem;
            margin: 0 0 0.2rem 0;
            font-weight: 800;
        }
        .dev-info .dev-role {
            color: #7EE0D4;
            font-family: 'Courier New', monospace;
            font-size: 0.95rem;
            margin-bottom: 1.1rem;
        }
        .dev-field {
            display: flex;
            gap: 0.6rem;
            margin-bottom: 0.45rem;
            font-size: 0.95rem;
        }
        .dev-field .dev-label {
            color: #02C39A;
            font-weight: 700;
            min-width: 110px;
            font-family: 'Courier New', monospace;
        }
        .dev-field .dev-value {
            color: #F7FAFA;
        }
        .dev-project-card {
            background-color: #F7FAFA;
            border: 1px solid #DCEBEA;
            border-left: 5px solid #02C39A;
            border-radius: 10px;
            padding: 1rem 1.3rem;
            margin-top: 0.5rem;
        }
        .dev-project-card .dev-project-label {
            color: #028090;
            font-weight: 700;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .dev-project-card .dev-project-title {
            color: #16282B;
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    import base64
    photo_path = BASE / "assets/developer_photo.jpg"
    photo_b64 = base64.b64encode(photo_path.read_bytes()).decode()

    st.markdown(f"""
    <div class="dev-card">
        <div class="dev-photo-wrap">
            <img src="data:image/jpeg;base64,{photo_b64}" />
        </div>
        <div class="dev-info">
            <h2>นางสาวเอมจิรา อ่วมเจริญ</h2>
            <div class="dev-role">Computer Science Student · NPRU</div>
            <div class="dev-field"><span class="dev-label">รหัสนักศึกษา</span><span class="dev-value">664245051</span></div>
            <div class="dev-field"><span class="dev-label">หมู่เรียน</span><span class="dev-value">66/44</span></div>
            <div class="dev-field"><span class="dev-label">สาขาวิชา</span><span class="dev-value">วิทยาการคอมพิวเตอร์ (Computer Science)</span></div>
            <div class="dev-field"><span class="dev-label">คณะ</span><span class="dev-value">มหาวิทยาลัยราชภัฏนครปฐม</span></div>
        </div>
    </div>
    <div class="dev-project-card">
        <div class="dev-project-label">โปรเจค</div>
        <div class="dev-project-title">₿ การพยากรณ์ทิศทางราคา Bitcoin ด้วย Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# หน้า 1: โจทย์ & Dataset
# =====================================================================
elif page == "1. โจทย์ & Dataset":
    st.title("1. การกำหนดปัญหาและ Dataset")

    st.markdown("""
    ### ปัญหาที่สนใจ
    ราคา Bitcoin มีความผันผวนสูงและซื้อขายกันตลอด 24 ชั่วโมง นักลงทุนรายย่อยจำนวนมาก
    ต้องการเครื่องมือช่วยประเมิน **แนวโน้มทิศทางราคาในวันถัดไป** (ขึ้น หรือ ลง)
    เพื่อประกอบการตัดสินใจเบื้องต้น จึงเลือกทำโจทย์นี้เป็น **ปัญหาการจำแนกประเภท (Binary Classification)**

    ### ทำไมเลือก Dataset ชุดนี้
    - เป็นข้อมูลราคา Bitcoin รายวันแบบ **OHLC (Open, High, Low, Close) + Volume** ย้อนหลังหลายปี
      ซึ่งเป็นข้อมูลมาตรฐานที่ใช้ในการวิเคราะห์ตลาดการเงินและงานวิจัยด้าน ML ทางการเงิน
    - มีจำนวนแถวเพียงพอ (>1,700 วัน) สำหรับการฝึกโมเดลและแบ่ง train/test แบบ time series
    - ข้อมูลอยู่ในรูปแบบ CSV ที่ต้องผ่านการทำความสะอาดจริง (comma, หน่วย K/M, %)
      ทำให้ได้ฝึกทักษะ Data Preprocessing ที่ใช้ได้จริง ไม่ใช่ข้อมูลสำเร็จรูป
    - สอดคล้องกับความสนใจส่วนตัวด้าน Cryptocurrency ที่เคยทำรายงานด้วย Power BI มาก่อน
    """)

    st.subheader("ตัวอย่างข้อมูลดิบ (หลังทำความสะอาดเบื้องต้น)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนแถวทั้งหมด", f"{len(raw_df):,}")
    c2.metric("ช่วงเวลา", f"{raw_df['Date'].min().date()} ถึง {raw_df['Date'].max().date()}")
    c3.metric("จำนวนคอลัมน์", len(raw_df.columns))

# =====================================================================
# หน้า 2: Data Preprocessing
# =====================================================================
elif page == "2. Data Preprocessing":
    st.title("2. Data Preprocessing")

    st.markdown("""
    ข้อมูลดิบจาก CSV มีปัญหาหลายจุดที่ต้องแก้ก่อนนำไปใช้กับโมเดล:

    | ปัญหาที่พบ | วิธีแก้ |
    |---|---|
    | ตัวเลขมี comma เช่น `"60,309.1"` | ลบ comma แล้วแปลงเป็น float |
    | Volume เป็นข้อความ เช่น `"106.30K"` | แปลงหน่วย K/M/B เป็นตัวเลขจริง |
    | Change % มีเครื่องหมาย `%` ติดอยู่ | ลบ `%` แล้วแปลงเป็นตัวเลข |
    | บางวันไม่มีข้อมูล Volume (missing value) | เติมด้วยค่ามัธยฐาน (median) |
    | ข้อมูลเรียงจากใหม่ไปเก่า | เรียงใหม่ตามวันที่ (เก่า → ใหม่) เพื่อทำ time series ได้ถูกต้อง |
    """)

    st.subheader("Feature ที่สร้างเพิ่ม (Feature Engineering)")
    st.markdown("""
    - **Daily_Return** — อัตราการเปลี่ยนแปลงราคาปิดเทียบวันก่อนหน้า (%)
    - **High_Low_Range** — ช่วงห่างของราคาสูงสุด/ต่ำสุดในแต่ละวัน (วัดความผันผวน)
    - **MA_7 / MA_21** — ค่าเฉลี่ยเคลื่อนที่ 7 และ 21 วัน (แนวโน้มระยะสั้น/กลาง)
    - **MA_Ratio** — อัตราส่วน MA_7/MA_21 บ่งบอกโมเมนตัมของแนวโน้ม
    - **Volatility_7** — ส่วนเบี่ยงเบนมาตรฐานของผลตอบแทนใน 7 วัน
    - **RSI_14** — Relative Strength Index รอบ 14 วัน (indicator มาตรฐานในการเทรด)
    - **Volume_Change** — การเปลี่ยนแปลงของปริมาณการซื้อขาย
    - **Target** — 1 ถ้าราคาปิดวันถัดไปสูงกว่าวันนี้, 0 ถ้าต่ำกว่าหรือเท่ากัน
    """)

    st.subheader("ข้อมูลหลังทำ Feature Engineering (พร้อมเข้าโมเดล)")
    st.dataframe(feat_df.head(10), use_container_width=True)

    st.subheader("สัดส่วนของ Target (ตรวจสอบ Class Balance)")
    target_counts = feat_df["Target"].value_counts().rename({0: "ลง (0)", 1: "ขึ้น (1)"})
    st.bar_chart(target_counts)

# =====================================================================
# หน้า 3: โมเดล ML
# =====================================================================
elif page == "3. โมเดล ML":
    st.title("3. การสร้างโมเดล Machine Learning")

    st.markdown("""
    โปรเจคนี้ทดลองใช้ 3 อัลกอริทึมสำหรับปัญหา Binary Classification เพื่อเปรียบเทียบประสิทธิภาพ:
    """)

    with st.expander("🔹 Logistic Regression", expanded=True):
        st.markdown("""
        โมเดลเชิงเส้นที่หาความสัมพันธ์ระหว่าง feature กับความน่าจะเป็นของ class (ขึ้น/ลง)
        ผ่านฟังก์ชัน sigmoid ข้อดีคือเรียบง่าย ตีความได้ (interpretable) และไม่ overfit ง่าย
        เหมาะเป็น **baseline model** สำหรับเปรียบเทียบกับโมเดลที่ซับซ้อนกว่า
        """)

    with st.expander("🔹 Random Forest"):
        st.markdown("""
        เป็น Ensemble Learning ที่รวมผลจาก **Decision Tree** จำนวนมาก (bagging) แต่ละต้นเรียนรู้
        จากข้อมูลและ feature ที่สุ่มมาบางส่วน แล้วนำผลโหวตของทุกต้นมาสรุปเป็นคำตอบสุดท้าย
        ช่วยลด overfitting เมื่อเทียบกับ Decision Tree เดี่ยว และบอก **Feature Importance** ได้
        """)

    with st.expander("🔹 Gradient Boosting"):
        st.markdown("""
        เป็น Ensemble Learning แบบ **boosting** ที่สร้างโมเดลทีละต้นตามลำดับ โดยแต่ละต้นใหม่จะ
        พยายามแก้ไข error ที่โมเดลก่อนหน้าทำนายผิด (เรียนรู้จาก residual) มักให้ความแม่นยำสูง
        แต่มีความเสี่ยง overfit มากกว่าถ้าปรับ hyperparameter ไม่เหมาะสม
        """)

    st.subheader("การตั้งค่าการทดลอง")
    c1, c2, c3 = st.columns(3)
    c1.metric("ข้อมูล Train", f"{meta['train_size']:,} วัน")
    c2.metric("ข้อมูล Test", f"{meta['test_size']:,} วัน")
    c3.metric("จำนวน Feature", len(FEATURES))
    st.caption("แบ่งข้อมูลตามลำดับเวลา (80% เก่าสุด = train, 20% ล่าสุด = test) "
               "ไม่สุ่มแบ่ง เพราะเป็นข้อมูล time series ที่ห้ามให้ข้อมูลอนาคตรั่วไหลเข้าไปใน training set")

    st.subheader("Feature ที่ใช้ในการฝึกโมเดล")
    st.code(", ".join(FEATURES))

# =====================================================================
# หน้า 4: เปรียบเทียบผลลัพธ์
# =====================================================================
elif page == "4. เปรียบเทียบผลลัพธ์":
    st.title("4. การประเมินและเปรียบเทียบโมเดล")

    st.subheader("ตารางเปรียบเทียบ Metric")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}",
            "Recall": "{:.2%}", "F1-Score": "{:.2%}"
        }).highlight_max(subset=["Accuracy"], color="#c8f7dc"),
        use_container_width=True,
    )
    best_row = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]
    st.success(f"โมเดลที่ดีที่สุด (Accuracy สูงสุด): **{best_row['Model']}** "
               f"— Accuracy {best_row['Accuracy']:.2%}")

    st.subheader("กราฟเปรียบเทียบ Metric ทั้ง 4 ตัว")
    st.image(str(BASE / "data/chart_model_comparison.png"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix (โมเดลที่ดีที่สุด)")
        st.image(str(BASE / "data/chart_confusion_matrix.png"), use_container_width=True)
    with col2:
        st.subheader("Feature Importance (Random Forest)")
        st.image(str(BASE / "data/chart_feature_importance.png"), use_container_width=True)

    st.info(
        "**หมายเหตุ:** Accuracy อยู่ที่ราว 50-53% ซึ่งใกล้เคียงการทายสุ่ม (50%) "
        "สะท้อนความเป็นจริงของตลาดการเงินที่มีประสิทธิภาพสูง (Efficient Market) "
        "ทำให้การทำนายทิศทางราคาระยะสั้นด้วย feature ทางเทคนิคอย่างเดียวเป็นเรื่องยากมาก "
        "แม้แต่ในงานวิจัยระดับมืออาชีพ — เป็นข้อสังเกตสำคัญที่ควรกล่าวถึงในการนำเสนอ",
        icon="📌",
    )

# =====================================================================
# หน้า 5: Live Demo
# =====================================================================
elif page == "5. ทำนายผล (Live Demo)":
    st.title("5. ทดลองใช้งานโมเดล (Live Demo)")
    st.markdown("ปรับค่าตัวแปรด้านล่างเพื่อดูว่าโมเดลทำนายว่าราคาพรุ่งนี้จะ **ขึ้น** หรือ **ลง**")

    last_row = feat_df.iloc[-1]

    st.caption(f"ค่าเริ่มต้นอ้างอิงจากข้อมูลวันล่าสุดในชุดข้อมูล ({last_row['Date'].date()})")

    col1, col2, col3 = st.columns(3)
    with col1:
        close = st.number_input("Close Price ($)", value=float(last_row["Close"]), step=100.0)
        open_p = st.number_input("Open Price ($)", value=float(last_row["Open"]), step=100.0)
        high = st.number_input("High Price ($)", value=float(last_row["High"]), step=100.0)
        low = st.number_input("Low Price ($)", value=float(last_row["Low"]), step=100.0)
    with col2:
        volume = st.number_input("Volume", value=float(last_row["Volume"]), step=1000.0)
        daily_return = st.slider("Daily Return (%)", -15.0, 15.0, float(last_row["Daily_Return"]))
        rsi = st.slider("RSI (14)", 0.0, 100.0, float(last_row["RSI_14"]))
        volume_change = st.slider("Volume Change (%)", -80.0, 200.0, float(last_row["Volume_Change"]))
    with col3:
        ma7 = st.number_input("MA 7 วัน", value=float(last_row["MA_7"]), step=100.0)
        ma21 = st.number_input("MA 21 วัน", value=float(last_row["MA_21"]), step=100.0)
        volatility = st.slider("Volatility (7 วัน)", 0.0, 15.0, float(last_row["Volatility_7"]))

    high_low_range = (high - low) / close * 100 if close else 0
    ma_ratio = ma7 / ma21 if ma21 else 1

    input_data = pd.DataFrame([{
        "Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume,
        "Daily_Return": daily_return, "High_Low_Range": high_low_range,
        "MA_7": ma7, "MA_21": ma21, "MA_Ratio": ma_ratio,
        "Volatility_7": volatility, "RSI_14": rsi, "Volume_Change": volume_change,
    }])[FEATURES]

    if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
        X_scaled = scaler.transform(input_data)

        st.divider()
        cols = st.columns(len(all_models))
        for col, (name, model) in zip(cols, all_models.items()):
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            with col:
                st.markdown(f"**{name}**")
                if pred == 1:
                    st.success(f"📈 ขึ้น ({proba[1]:.1%})")
                else:
                    st.error(f"📉 ลง ({proba[0]:.1%})")

        st.caption(
            "⚠️ ผลการทำนายนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน "
            "ราคา Cryptocurrency มีความผันผวนสูงและมีปัจจัยภายนอกจำนวนมากที่โมเดลนี้ไม่ได้นำมาพิจารณา"
        )

st.sidebar.divider()
st.sidebar.caption("โปรเจค BI/Data Warehouse — Nakhon Pathom Rajabhat University")
