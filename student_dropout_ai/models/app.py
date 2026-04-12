import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os
import plotly.express as px
import shap   # ✅ ADDED

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dropout AI", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        return load("../saved_models/ensemble_model.joblib")
    except:
        return None

model = load_model()

# =========================
# SHAP FUNCTION
# =========================
def get_shap_values(model, X):
    try:

        # Handle VotingClassifier
        if hasattr(model, "estimators_"):
            base_model = None

            for est in model.estimators_:
                if "RandomForest" in str(type(est)):
                    base_model = est
                    break

            if base_model is None:
                st.warning("No supported model found for SHAP")
                return None
        else:
            base_model = model

        # Try TreeExplainer first
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)
        except:
            explainer = shap.Explainer(base_model)
            shap_values = explainer(X)

        # Fix classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        return shap_values

    except Exception as e:
        st.warning(f"SHAP Error: {e}")
        return None
    

# =========================
# PREPROCESS
# =========================
def preprocess(df):
    df = df.copy()

    cols = [
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (evaluations)',
        'Tuition fees up to date',
        'Debtor',
        'Unemployment rate'
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = 0

    df['grade_momentum'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']

    df['eval_efficiency'] = df['Curricular units 2nd sem (approved)'] / (
        df['Curricular units 2nd sem (evaluations)'].replace(0, 1)
    )

    df['socio_economic_risk'] = (
        df['Debtor'] +
        (1 - df['Tuition fees up to date']) +
        df['Unemployment rate'] / 20
    )

    df = df.fillna(0)
    df.columns = df.columns.astype(str)

    return df

# =========================
# ALIGN FEATURES
# =========================
def align(df):
    if model is None:
        return df

    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    df.columns = df.columns.astype(str)

    return df

# =========================
# REASON & SUGGESTION
# =========================
def generate_reason(row):
    r = []
    if row['Curricular units 2nd sem (approved)'] < 3:
        r.append("Low subject clearance")
    if row['Curricular units 2nd sem (grade)'] < 10:
        r.append("Low academic performance")
    if row['Tuition fees up to date'] == 0:
        r.append("Fees pending")
    if row['Debtor'] == 1:
        r.append("Financial debt")
    if row['Unemployment rate'] > 10:
        r.append("High unemployment")
    return ", ".join(r) if r else "Stable"

def generate_suggestion(row):
    s = []
    if row['Curricular units 2nd sem (grade)'] < 10:
        s.append("Improve academics")
    if row['Tuition fees up to date'] == 0:
        s.append("Clear fees")
    if row['Debtor'] == 1:
        s.append("Apply for scholarship")
    if row['Curricular units 2nd sem (approved)'] < 3:
        s.append("Focus on passing subjects")
    return ", ".join(s) if s else "Maintain performance"

# =========================
# PDF GENERATION
# =========================
def generate_pdf(df):
    file_path = "student_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Student Dropout Prediction Report", styles['Title']))
    elements.append(Spacer(1, 20))

    pie_path = "pie.png"
    counts = df['Prediction'].value_counts()
    plt.figure()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.savefig(pie_path)
    plt.close()

    if os.path.exists(pie_path):
        elements.append(Image(pie_path, width=300, height=300))

    bar_path = "bar.png"
    plt.figure()
    counts.plot(kind='bar')
    plt.savefig(bar_path)
    plt.close()

    elements.append(Image(bar_path, width=400, height=300))

    trend_path = "trend.png"
    plt.figure()
    df['Confidence'].reset_index(drop=True).plot()
    plt.savefig(trend_path)
    plt.close()

    elements.append(Image(trend_path, width=400, height=300))

    elements.append(Spacer(1, 20))

    for _, row in df.iterrows():
        color = "red" if row['Prediction']=="Dropout" else "green"

        text = f"""
        <b>Student:</b> {row['Student Name']}<br/>
        <b>Status:</b> <font color='{color}'>{row['Prediction']}</font><br/>
        <b>Reason:</b> {row['Reason']}<br/>
        <b>Suggestion:</b> {row['Suggestion']}<br/>
        <b>Confidence:</b> {round(row['Confidence']*100,2)}%<br/>
        <br/>----------------------------<br/>
        """

        elements.append(Paragraph(text, styles['Normal']))
        elements.append(Spacer(1, 10))

    doc.build(elements)

    for f in [pie_path, bar_path, trend_path]:
        if os.path.exists(f):
            os.remove(f)

    return file_path

# =========================
# SESSION
# =========================
if "data" not in st.session_state:
    st.session_state.data = None

# =========================
# UI
# =========================
st.title("🎓 Student Dropout AI System")

tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])

# =========================
# SINGLE
# =========================
with tab1:
    sem1 = st.slider("Sem1 Grade", 0.0, 20.0, 10.0)
    sem2 = st.slider("Sem2 Grade", 0.0, 20.0, 10.0)
    approved = st.slider("Approved Subjects", 0, 10, 5)
    evals = st.slider("Evaluations", 0, 10, 5)
    fees = st.selectbox("Fees Paid", [1,0])
    debtor = st.selectbox("Debtor", [0,1])
    unemployment = st.slider("Unemployment",0.0,20.0,8.0)

    if st.button("Predict Single"):
        raw_df = pd.DataFrame([{
        'Curricular units 1st sem (grade)': sem1,
        'Curricular units 2nd sem (grade)': sem2,
        'Curricular units 2nd sem (approved)': approved,
        'Curricular units 2nd sem (evaluations)': evals,
        'Tuition fees up to date': fees,
        'Debtor': debtor,
        'Unemployment rate': unemployment
        }])

        df_proc = preprocess(raw_df)
        df_model = align(df_proc).astype(float)

        pred = model.predict(df_model)[0]
        prob = model.predict_proba(df_model)[0]

        confidence = max(prob)
        label = "Dropout" if pred == 0 else "Graduate"

        if label == "Dropout":
            st.error(f"🔴 Dropout Risk ({confidence*100:.2f}%)")
        else:
            st.success(f"🟢 Likely Graduate ({confidence*100:.2f}%)")
        
        # 🎯 Risk Meter
        st.subheader("🎯 Risk Score")
        st.progress(float(confidence))

        if label == "Dropout":
            st.markdown(f"### 🔴 {confidence*100:.1f}% Risk")
        else:
            st.markdown(f"### 🟢 {confidence*100:.1f}% Safe")

        reason = generate_reason(df_proc.iloc[0])
        suggestion = generate_suggestion(df_proc.iloc[0])

        st.subheader("📌 Reason")
        st.info(reason)

        st.subheader("💡 Suggestion")
        st.success(suggestion)

        # ✅ SHAP SINGLE (PREMIUM UI)
        st.subheader("🧠 SHAP Explainability")
        shap_values = get_shap_values(model, df_model)

        if shap_values is not None:
            shap_array = np.array(shap_values)
            
            if shap_array.ndim == 3:
                shap_array = shap_array[:, :, 1]

            shap_df = pd.DataFrame(shap_array, columns=df_model.columns)

            top_n = 8
            row = shap_df.iloc[0]
            top_features = row.abs().sort_values(ascending=False).head(top_n).index
            plot_df = row[top_features].sort_values()

            colors = ["red" if v > 0 else "green" for v in plot_df]

            fig, ax = plt.subplots(figsize=(7, 4))
            plot_df.plot(kind="barh", ax=ax, color=colors)

            ax.set_title("🧠 What is affecting this prediction?")
            ax.set_xlabel("Impact")

            st.pyplot(fig)

            # Insights
            st.subheader("📊 Key Insights")

            risk = plot_df[plot_df > 0].sort_values(ascending=False).head(3)
            safe = plot_df[plot_df < 0].sort_values().head(3)

            col1, col2 = st.columns(2)

            with col1:
                st.error("🔴 Increasing Dropout Risk")
                for f in risk.index:
                    st.write(f"• {f}")

            with col2:
                st.success("🟢 Supporting Graduation")
                for f in safe.index:
                    st.write(f"• {f}")

# =========================
# BULK
# =========================
with tab2:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "Student Name" not in df.columns:
            df["Student Name"] = ["Student_"+str(i+1) for i in range(len(df))]

        if st.button("Predict"):
            df_proc = preprocess(df)
            df_model = align(df_proc)

            pred = model.predict(df_model)
            prob = model.predict_proba(df_model)

            df['Prediction'] = np.where(pred==0,"Dropout","Graduate")
            df['Confidence'] = prob.max(axis=1)
            df['Reason'] = df_proc.apply(generate_reason,axis=1)
            df['Suggestion'] = df_proc.apply(generate_suggestion,axis=1)

            def risk_level(conf):
                if conf > 0.75:
                    return "High Risk"
                elif conf > 0.5:
                    return "Medium Risk"
                else:
                    return "Low Risk"

            df['Risk Level'] = df['Confidence'].apply(risk_level)

            st.session_state.data = df
            st.session_state.df_model = df_model  # ✅ store for SHAP

    if st.session_state.data is not None:
        df = st.session_state.data
        df_model = st.session_state.df_model   

        p = st.multiselect("Prediction",["Dropout","Graduate"],["Dropout","Graduate"])
        c = st.slider("Confidence",0.0,1.0,0.0)

        # 🔍 Search
        search = st.text_input("🔍 Search Student")

        fdf = df[(df['Prediction'].isin(p)) & (df['Confidence']>=c)]

        # 🔽 Sort by highest risk first
        fdf = fdf.sort_values(by="Confidence", ascending=False)

        if search:
            fdf = fdf[fdf.apply(lambda row: search.lower() in str(row).lower(), axis=1)]

        counts = fdf['Prediction'].value_counts()

        fig = px.pie(fdf,names="Prediction",title="🎯 Dropout vs Graduate Distribution",hole=0.4)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.histogram(
                  fdf,
                  x="Confidence",
                  color="Prediction",
                  nbins=20,
                  title="📊 Confidence Distribution")
        
        fig2.update_layout(height=500)

        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
                    fdf,
                    x="Risk Level",
                    color="Prediction",
                    title="🚦 Risk Level Distribution",
                    barmode="group"
                )

        st.plotly_chart(fig3, use_container_width=True)

        top10 = fdf.sort_values(by="Confidence", ascending=False).head(10)

        fig4 = px.bar(
                     top10,
                     x="Confidence",
                     y="Student Name",
                     orientation="h",
                     color="Prediction",
                     title="🚨 Top 10 Risky Students"
                    )
        fig4.update_layout(height=600)

        st.plotly_chart(fig4, use_container_width=True)
        
        
        # ✅ SHAP BULK (PREMIUM UI)
        st.subheader("🧠 SHAP (First Student)")
        shap_values = get_shap_values(model, df_model.iloc[[0]])

        if shap_values is not None:
            shap_array = np.array(shap_values)

            if shap_array.ndim == 3:
                shap_array = shap_array[:, :, 1]

            shap_df = pd.DataFrame(shap_array, columns=df_model.columns)

            top_n = 8
            row = shap_df.iloc[0]
            top_features = row.abs().sort_values(ascending=False).head(top_n).index
            plot_df = row[top_features].sort_values()

            colors = ["red" if v > 0 else "green" for v in plot_df]

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            plot_df.plot(kind="barh", ax=ax2, color=colors)

            ax2.set_title("🧠 What is affecting this prediction?")
            ax2.set_xlabel("Impact")

            st.pyplot(fig2)

            # Insights
            st.subheader("📊 Key Insights")

            risk = plot_df[plot_df > 0].sort_values(ascending=False).head(3)
            safe = plot_df[plot_df < 0].sort_values().head(3)

            col1, col2 = st.columns(2)

            with col1:
                st.error("🔴 Increasing Dropout Risk")
                for f in risk.index:
                    st.write(f"• {f}")
            

            with col2:
                st.success("🟢 Supporting Graduation")
                for f in safe.index:
                    st.write(f"• {f}")

        for _, row in fdf.iterrows():
            badge = "🔴" if row['Prediction']=="Dropout" else "🟢"
            with st.expander(f"{badge} {row['Student Name']}"):
                st.write("Reason:", row['Reason'])
                st.write("Suggestion:", row['Suggestion'])
                st.write("Risk Level:", row['Risk Level'])
        
        # 📥 CSV Download
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download CSV",
            csv,
            "students.csv",
            "text/csv"
        )

        pdf_path = generate_pdf(df)

        with open(pdf_path,"rb") as f:
            st.download_button("📄 Download PDF", f, "report.pdf")



