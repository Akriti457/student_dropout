import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="Student Dropout AI", layout="wide")
st.title("🎓 AI Student Dropout Prediction System")

# =======================
# AUTH SYSTEM (NEW)
# =======================
USERS_FILE = "users.csv"

if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_user(username, password):
    df = load_users()

    username = username.strip()
    password = password.strip()

    if username in df["username"].astype(str).str.strip().values:
        return False

    new_row = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(USERS_FILE, index=False)
    return True

def authenticate(username, password):
    df = load_users()

    username = username.strip()
    password = password.strip()

    df["username"] = df["username"].astype(str).str.strip()
    df["password"] = df["password"].astype(str).str.strip()

    user = df[
        (df["username"].str.lower() == username.lower()) &
        (df["password"] == password)
    ]

    return not user.empty

# session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# =======================
# LOGIN / SIGNUP UI
# =======================
if not st.session_state.logged_in:

    menu = st.radio("Choose Action", ["Login", "Signup"])

    if menu == "Signup":
        st.subheader("Create Account")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Signup"):
            if save_user(new_user, new_pass):
                st.success("Account created! Please login.")
            else:
                st.error("User already exists!")

    else:
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()   # 🚨 STOP APP until login

# =======================
# LOGOUT BUTTON
# =======================
st.sidebar.success(f"Logged in as: {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# =======================
# LOAD MODEL
# =======================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()
model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessing"]

# =======================
# HISTORY
# =======================
HISTORY_FILE = "history.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Timestamp", "Prediction", "Confidence"]).to_csv(HISTORY_FILE, index=False)

def save_history(pred, conf):
    df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": pred,
        "Confidence": conf,
        "User": st.session_state.get("user", "unknown")
    }])

    df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False
    )

def load_history():
    try:
        df = pd.read_csv(HISTORY_FILE)

        # force correct types (VERY IMPORTANT)
        if "Confidence" in df.columns:
            df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")

        return df

    except Exception:
        # if file is corrupted → reset safely
        df = pd.DataFrame(columns=["Timestamp", "Prediction", "Confidence", "User"])
        df.to_csv(HISTORY_FILE, index=False)
        return df

# =======================
# PDF FUNCTIONS (UNCHANGED)
# =======================
def generate_pdf(label, confidence):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Student Dropout Report", styles["Title"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Prediction: {label}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

def generate_batch_pdf(df):
    doc = SimpleDocTemplate("batch_report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Batch Prediction Report", styles["Title"]))
    content.append(Spacer(1, 20))

    for i, row in df.iterrows():
        content.append(Paragraph(f"Student {i+1}", styles["Heading3"]))
        content.append(Paragraph(f"Prediction: {row['Prediction']}", styles["Normal"]))
        content.append(Paragraph(f"Dropout Probability: {row['Dropout_Prob']:.2f}", styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)

    with open("batch_report.pdf", "rb") as f:
        return f.read()

# =======================
# FEATURE ENGINEERING (UNCHANGED)
# =======================
def prepare_input(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in ["Internet_Access", "Scholarship", "Part_Time_Job"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if "Department" in df.columns and "Semester" in df.columns:
        df["Dept_Semester"] = df["Department"] + "_" + df["Semester"]

    df["Income_Stress_Ratio"] = df["Stress_Index"] / (df["Family_Income"] + 1)
    df["GPA_vs_CGPA"] = df["GPA"] - df["CGPA"]
    df["GPA_vs_SGPA"] = df["GPA"] - df["Semester_GPA"]

    df["Job_Stress_Interaction"] = df["Part_Time_Job"] * df["Stress_Index"]

    df["Student_Pressure_Index"] = (
        df["Assignment_Delay_Days"] + df["Travel_Time_Minutes"]
        + df["Part_Time_Job"] + df["Stress_Index"]
    ) / (df["Attendance_Rate"] + df["Study_Hours_per_Day"] + 0.01)

    df["Academic_Risk"] = (
        (df["Attendance_Rate"] < 75).astype(int)
        + (df["Study_Hours_per_Day"] < 2).astype(int)
        + (df["Assignment_Delay_Days"] > 5).astype(int)
    )

    drop_cols = [c for c in ["Department", "Semester"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df

# =======================
# MAIN APP (YOUR ORIGINAL UI KEPT)
# =======================
st.markdown("## 🧾 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 15, 30, 21)
    gpa = st.number_input("GPA", 0.0, 10.0, 2.8)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 2.6)
    sgpa = st.number_input("Semester GPA", 0.0, 10.0, 2.5)

with col2:
    study_hours = st.slider("Study Hours/Day", 0, 12, 3)
    delay = st.slider("Assignment Delay", 0, 15, 4)
    attendance = st.slider("Attendance", 0, 100, 70)
    travel = st.slider("Travel Time", 0, 120, 30)

with col3:
    income = st.number_input("Family Income", 30000)
    stress = st.slider("Stress Index", 0.0, 10.0, 6.5)

    internet = st.selectbox("Internet", ["Yes", "No"])
    scholarship = st.selectbox("Scholarship", ["Yes", "No"])
    job = st.selectbox("Part-Time Job", ["Yes", "No"])

col4, col5 = st.columns(2)

with col4:
    semester = st.selectbox("Semester", ["Year 1", "Year 2", "Year 3"])

with col5:
    department = st.selectbox("Department", ["Engineering", "Science", "Arts"])
    parent = st.selectbox("Parental Education", ["High School", "Graduate", "Postgraduate", "Unknown"])

# =======================
# PREDICTION
# =======================
if st.button("🚀 Predict Now"):

    data = {
        "Age": age,
        "GPA": gpa,
        "CGPA": cgpa,
        "Semester_GPA": sgpa,
        "Study_Hours_per_Day": study_hours,
        "Assignment_Delay_Days": delay,
        "Attendance_Rate": attendance,
        "Travel_Time_Minutes": travel,
        "Family_Income": income,
        "Internet_Access": internet,
        "Scholarship": scholarship,
        "Part_Time_Job": job,
        "Stress_Index": stress,
        "Semester": semester,
        "Department": department,
        "Parental_Education": parent
    }

    input_df = pd.DataFrame([data])
    processed = prepare_input(input_df)

    pred = pipeline.predict(processed)[0]
    proba = pipeline.predict_proba(processed)[0]
    conf = proba[pred] * 100

    label = "Dropout" if pred == 1 else "Persist"

    save_history(label, conf)

    st.success(f"{label} ({conf:.2f}%)")

    pdf = generate_pdf(label, conf)
    st.download_button("Download Report", pdf, "report.pdf")

# =======================
# DASHBOARD (UNCHANGED)
# =======================
st.markdown("---")
st.subheader("Dashboard")

history = load_history()

if not history.empty:
    st.metric("Total Predictions", len(history))
    st.metric("Dropout Rate", f"{(history['Prediction']=='Dropout').mean()*100:.1f}%")
    avg_conf = pd.to_numeric(history["Confidence"], errors="coerce").mean()
    st.metric( "Avg Confidence",f"{avg_conf:.1f}%" if pd.notna(avg_conf) else "0.0%")
    st.line_chart(history["Confidence"])
    st.bar_chart(history["Prediction"].value_counts())
    st.dataframe(history.tail(10))

# =======================
# BATCH UPLOAD (UNCHANGED)
# =======================
st.markdown("---")
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    processed = prepare_input(df)

    preds = pipeline.predict(processed)
    probs = pipeline.predict_proba(processed)

    df["Prediction"] = ["Dropout" if p == 1 else "Persist" for p in preds]
    df["Dropout_Prob"] = probs[:, 1]

    st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), "results.csv")

    pdf = generate_batch_pdf(df)
    st.download_button("Download PDF", pdf, "batch_report.pdf", mime="application/pdf")
