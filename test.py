import pandas as pd
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------------------
# Туслах функцууд
# -------------------
def parse_price(price_str):
    if not price_str or pd.isnull(price_str):
        return None
    price_str = str(price_str).replace(",", "").replace("₮", "").strip().lower()
    multiplier = 1
    if "сая" in price_str:
        multiplier = 1_000_000
        price_str = price_str.replace("сая", "").strip()
    elif "төгрөг" in price_str:
        multiplier = 1
        price_str = price_str.replace("төгрөг", "").strip()
    match = re.search(r"(\d+(\.\d+)?)", price_str)
    if match:
        return float(match.group(1)) * multiplier
    return None

def parse_area(area_str):
    if not area_str or pd.isnull(area_str):
        return None
    match = re.search(r"(\d+(\.\d+)?)\s*м[2²]?", str(area_str).lower())
    if match:
        return float(match.group(1))
    return None

def parse_int(value):
    """Текстээс зөвхөн тоон утгыг гаргаж int болгох"""
    if pd.isnull(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None

# -------------------
# Streamlit апп
# -------------------
st.title("🏠 Орон сууцны үнэ таамаглагч (ML)")

uploaded_file = st.file_uploader("📂 CSV файл оруулна уу", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Үнэ багана автоматаар олох
    if "price_numeric" in df.columns:
        pass
    elif "price_text" in df.columns:
        df["price_numeric"] = df["price_text"].apply(parse_price)
    else:
        price_col = None
        for col in df.columns:
            if "үнэ" in col.lower() or "price" in col.lower():
                price_col = col
                break
        if price_col:
            df["price_numeric"] = df[price_col].apply(parse_price)
        else:
            st.error("⚠️ Үнийн багана олдсонгүй!")
            st.stop()

    # Талбай автоматаар олох
    if "area_numeric" in df.columns:
        pass
    elif "Талбай" in df.columns:
        df["area_numeric"] = df["Талбай"].apply(parse_area)
    else:
        st.error("⚠️ Талбайн багана олдсонгүй!")
        st.stop()

    # Тоон болгож болох багануудыг parse_int ашиглаж хөрвүүлэх
    for col in ["Барилгын давхар", "Цонхны тоо", "Ашиглалтанд орсон он", "Хэдэн давхарт"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_int)

    # Хоосон утгууд хасах
    df = df.dropna(subset=["price_numeric", "area_numeric"])

    # ML оролт, гаралт
    feature_cols = [
        "district", "Шал", "Тагт", "Ашиглалтанд орсон он", "Гараж",
        "Цонх", "Барилгын давхар", "Хаалга", "area_numeric",
        "Хэдэн давхарт", "Цонхны тоо", "Барилгын явц", "Цахилгаан шаттай эсэх"
    ]
    # Байхгүй багануудыг хасах
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df["price_numeric"]

    # Категори, тоон баганууд ялгах
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # ML pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Сургалт
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.info(f"📊 ML загварын R² score: {score:.2f}")

    # -------------------
    # Таамаглал хийх UI
    # -------------------
    st.subheader("🔮 Үнэ таамаглуулах")

    input_data = {}
    for col in feature_cols:
        if col in categorical_cols:
            options = list(df[col].dropna().unique())
            input_data[col] = st.selectbox(f"{col}:", options)
        else:
            default_value = int(df[col].dropna().mean()) if not df[col].dropna().empty else 0.0
            input_data[col] = st.number_input(f"{col}:", value=default_value)

    if st.button("💰 Таамаглах"):
        prediction = model.predict(pd.DataFrame([input_data]))[0]
        st.success(f"💰 ML таамагласан үнэ: {prediction:,.0f} ₮")

else:
    st.warning("📂 Эхлээд CSV файл оруулна уу.")
