# ------------------------
# app.py
# ------------------------
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import cloudinary
import cloudinary.uploader

st.set_page_config(page_title="Schul-Fundbörse", layout="wide")

# ------------------------
# Cloudinary Konfiguration (SICHER)
# ------------------------
cloudinary.config(
    cloud_name=os.getenv("ducsrjpz9"),
    api_key=os.getenv("633744854218788"),
    api_secret=os.getenv("DeHZyqChMgUGdt7UEooyXVrsR2s")
)

# ------------------------
# Modelle laden
# ------------------------
model_clothes = load_model("keras_Model_clothes.h5", compile=False)
with open("labels_clothes.txt", "r") as f:
    class_names_clothes = f.readlines()

model_colors = load_model("keras_Model_colors.h5", compile=False)
with open("labels_colors.txt", "r") as f:
    class_names_colors = f.readlines()

# ------------------------
# Kategorien
# ------------------------
haupt_kategorien = ["T-Shirt", "Pullover", "Brille"]
farbe_kategorien = ["Blau", "Rot", "Schwarz", "Weiß", "Grün", "Grau", "Braun", "Sonstige"]

# ------------------------
# CSV-Datei
# ------------------------
DATA_FILE = "fundboerse.csv"
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["Typ","Beschreibung","Ort","Foto","Datum"])
    df.to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)

# ------------------------
# Vorhersagefunktionen
# ------------------------
def predict_clothes(img: Image.Image):
    size = (224, 224)
    image_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32)/127.5)-1
    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model_clothes.predict(data)
    index = np.argmax(prediction)
    class_name = class_names_clothes[index].strip()

    mapping = {
        "0 Brille": "Brille",
        "glasses": "Brille",
        "t-shirt": "T-Shirt",
        "Tshirt": "T-Shirt",
        "pullover": "Pullover",
        "sweater": "Pullover"
    }

    class_name = mapping.get(class_name, class_name)
    confidence_score = prediction[0][index]
    return class_name, confidence_score


def predict_color(img: Image.Image):
    size = (224, 224)
    image_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32)/127.5)-1
    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model_colors.predict(data)
    index = np.argmax(prediction)
    class_name = class_names_colors[index].strip()

    color_class = class_name[2:].strip() if len(class_name) > 2 else class_name

    confidence_score = prediction[0][index]
    if confidence_score < 0.6:
        color_class = "Sonstige"

    return color_class, confidence_score


# ------------------------
# Speichern
# ------------------------
def save_entry(typ, beschreibung, ort, foto_url):
    global df
    datum = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "Typ": typ,
        "Beschreibung": beschreibung,
        "Ort": ort,
        "Foto": foto_url,
        "Datum": datum
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    st.success(f"Gegenstand '{typ}' erfolgreich gemeldet!")


# ------------------------
# Layout
# ------------------------
st.title("🏫 Schul-Fundbörse")

menu = ["Gegenstand melden", "Fundstücke durchsuchen", "Galerie"]
choice = st.sidebar.selectbox("Menü", menu)

# ------------------------
# MELDEN
# ------------------------
if choice == "Gegenstand melden":

    st.subheader("📝 Gegenstand melden")

    with st.form("fund_form"):

        uploaded_file = st.file_uploader("Foto hochladen", type=["jpg","jpeg","png"])
        beschreibung = st.text_area("Beschreibung")
        ort = st.text_input("Fundort / Ort")
        submit = st.form_submit_button("Melden")

        if submit:

            if uploaded_file is not None:

                img = Image.open(uploaded_file).convert("RGB")

                # KI
                item_class, conf_item = predict_clothes(img)
                if item_class not in haupt_kategorien:
                    item_class = "Sonstige"

                color_class, conf_color = predict_color(img)
                if color_class not in farbe_kategorien:
                    color_class = "Sonstige"

                typ = f"{color_class} {item_class}"

                # ☁️ Upload zu Cloudinary
                upload_result = cloudinary.uploader.upload(
                    uploaded_file,
                    folder="schul_fundboerse",
                    resource_type="image"
                )

                foto_url = upload_result["secure_url"]

                # Speichern
                save_entry(typ, beschreibung, ort, foto_url)

                # Anzeige
                st.image(foto_url, width=250)
                st.write(f"Erkannt: {typ}")
                st.write(f"Kleidungsstück Confidence: {conf_item:.2f}")
                st.write(f"Farbe Confidence: {conf_color:.2f}")

            else:
                st.warning("Bitte ein Foto hochladen!")

# ------------------------
# DURCHSUCHEN
# ------------------------
elif choice == "Fundstücke durchsuchen":

    st.subheader("🔍 Fundstücke durchsuchen")

    suche = st.text_input("Suchbegriff (Beschreibung oder Typ)")
    ergebnisse = df.copy()

    if suche:
        ergebnisse = ergebnisse[
            ergebnisse["Beschreibung"].str.contains(suche, case=False, na=False) |
            ergebnisse["Typ"].str.contains(suche, case=False, na=False)
        ]

    st.write(f"{len(ergebnisse)} Ergebnis(se) gefunden:")

    for _, row in ergebnisse.iterrows():
        st.markdown(f"**{row['Typ']}** - {row['Ort']} ({row['Datum']})")
        st.write(row["Beschreibung"])
        if row["Foto"]:
            st.image(row["Foto"], width=200)
        st.write("---")


# ------------------------
# GALERIE
# ------------------------
elif choice == "Galerie":

    st.subheader("📸 Galerie aller Fundstücke")

    # Filter
    filter_typ = st.selectbox("Filter nach Typ", ["Alle"] + haupt_kategorien)
    filter_farbe = st.selectbox("Filter nach Farbe", ["Alle"] + farbe_kategorien)

    ergebnisse = df.copy()

    if filter_typ != "Alle":
        ergebnisse = ergebnisse[ergebnisse["Typ"].str.contains(filter_typ, case=False, na=False)]

    if filter_farbe != "Alle":
        ergebnisse = ergebnisse[ergebnisse["Typ"].str.contains(filter_farbe, case=False, na=False)]

    st.write(f"{len(ergebnisse)} Ergebnis(se) gefunden:")

    # Bilder im Grid anzeigen (3 Spalten)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(ergebnisse.iterrows()):
        with cols[idx % 3]:
            st.image(row["Foto"], use_column_width=True)
            st.markdown(f"**{row['Typ']}**")
            st.write(row["Ort"])
