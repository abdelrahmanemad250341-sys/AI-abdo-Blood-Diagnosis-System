import os
import csv
import numpy as np
import base64
import uuid
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# إعداد فولدر رفع الصور
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# تحميل الموديل
MODEL_PATH = "ultra_blood_model.h5"
print("========================================")
print("جاري تحميل الموديل... الرجاء الانتظار.")
print("========================================")
model = load_model(MODEL_PATH)
print("تم تحميل الموديل بنجاح! السيرفر جاهز للعمل.")

CLASSES = ["Anemia", "Leukemia", "Malaria", "Normal"]


# دالة لقراءة بيانات المرضى من ملف الإكسيل
def get_all_patients():
    patients = []
    csv_file = "patients_database.csv"
    if os.path.isfile(csv_file):
        with open(csv_file, mode="r", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)  # تخطي صف العناوين
            for row in reader:
                if row:  # عشان نتجنب الصفوف الفاضية
                    patients.append(row)
    # عكس القائمة عشان أحدث مريض يظهر فوق
    return patients[::-1]


def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    confidence = round(np.max(predictions) * 100, 2)

    probabilities = {
        CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))
    }
    return CLASSES[class_index], confidence, probabilities


@app.route("/", methods=["GET"])
def home():
    # نبعت بيانات المرضى للصفحة الرئيسية
    return render_template("index.html", patients=get_all_patients())


@app.route("/save_patient", methods=["POST"])
def save_patient():
    if request.method == "POST":
        name = request.form.get("patient_name")
        age = request.form.get("patient_age")
        gender = request.form.get("patient_gender")
        phone = request.form.get("patient_phone")
        blood_type = request.form.get("blood_type")
        history = request.form.get("medical_history")

        csv_file = "patients_database.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(
                    ["Name", "Age", "Gender", "Phone", "Blood Type", "History"]
                )
            writer.writerow([name, age, gender, phone, blood_type, history])

        success_message = f"Patient Data for ({name}) Saved Successfully!"
        return render_template(
            "index.html", success_msg=success_message, patients=get_all_patients()
        )


@app.route("/predict", methods=["POST"])
def predict():
    filepath = None

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

    elif "camera_image" in request.form and request.form["camera_image"] != "":
        img_data = request.form["camera_image"]
        img_data = img_data.split(",")[1]
        filename = f"capture_{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        with open(filepath, "wb") as fh:
            fh.write(base64.b64decode(img_data))

    if filepath:
        disease_name, conf, probs = predict_disease(filepath)
        img_url = f"/{filepath}"

        return render_template(
            "index.html",
            prediction=disease_name,
            confidence=conf,
            probabilities=probs,
            image_path=img_url,
            patients=get_all_patients(),  # نبعت البيانات هنا كمان
        )

    return render_template("index.html", patients=get_all_patients())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
