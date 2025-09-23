import streamlit as st
import ML_models as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

# ---------------- Page config ---------------- #
st.set_page_config(
    page_title="PhishDetect Content-based Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)

disable_warnings(InsecureRequestWarning)

# ---------------- Top area ---------------- #
st.markdown("""
# 🛡️ PhishDetect  Content-based Phishing Website Detector
_A lightweight educational app that classifies webpages as **legitimate** or **phishing** using content-only features (HTML)._

> **Note:** This app uses scikit-learn models from `ML_models.py` and focuses only on page content not URL heuristics.
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    st.image("	https://prop.idcheck.tech/assets/img/fraud-detection.png",
    caption="Phishing Website Detection",
    use_container_width=True,width=140)
    st.markdown("**Quick tips**")
    st.markdown(
        "- Phishing pages are often simple and ask for credentials\n"
        "- Pages can change fast; results are educational\n"
        "- Use the examples to try live checks"
    )
    st.info("Paste the full http(s) URL. Fetching may take 3–5 seconds.")

with col2:
    st.header("How it works")
    st.write(""" The detector follows a simple **3-step pipeline** to analyze webpages:

    1. **Fetch the page**  
       The system downloads the full HTML of the target URL using a web request.

    2. **Extract meaningful features**  
       The HTML is parsed with BeautifulSoup to identify elements such as:  
       - Number of input fields (login forms, password boxes, etc.)  
       - Scripts and embedded objects  
       - Links and redirections  
       - Presence of suspicious keywords (e.g., *verify, password, account*)  

    3. **Classify using trained ML models**  
       The extracted features are transformed into numerical vectors and passed to a selected machine learning model (Naive Bayes, SVM, Random Forest, etc.).  
       - The model predicts whether the page is **legitimate** or **phishing**.  
       - You can view the raw probabilities for more detail.

    """)

# ---------------- Project details expander ---------------- #
with st.expander("Project details & dataset", expanded=False):
    st.subheader("Approach")
    st.write("Supervised learning on engineered content features from HTML. "
             "Data sources: PhishTank and Tranco.")

    st.subheader("Dataset snapshot")
    try:
        phishing_count = ml.phishing_df.shape[0]
        legit_count = ml.legitimate_df.shape[0]
        st.write(f"Phishing: **{phishing_count}**, Legitimate: **{legit_count}**, "
                 f"Total: **{phishing_count + legit_count}**")

        labels = ['Phishing', 'Legitimate']
        sizes = [phishing_count, legit_count]
        explode = (0.08, 0)
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.markdown("**Feature table (first 10 rows)**")
        st.dataframe(ml.df.head(10))

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(ml.df)
        st.download_button("Download structured dataset (CSV)", csv,
                           "phishing_structured_data.csv", "text/csv")

    except Exception as e:
        st.warning(f"Dataset preview not available: {e}")

# ---------------- Sidebar ---------------- #
st.sidebar.header("Model & Options")
choice = st.sidebar.selectbox("Select model", [
    'Gaussian Naive Bayes', 'Support Vector Machine',
    'Decision Tree', 'Random Forest',
    'AdaBoost', 'Neural Network', 'K-Neighbours'
])

st.sidebar.markdown("---")
show_probs = st.sidebar.checkbox("Show raw model output (if available)", value=False)

model_map = {
    'Gaussian Naive Bayes': ml.nb_model,
    'Support Vector Machine': ml.svm_model,
    'Decision Tree': ml.dt_model,
    'Random Forest': ml.rf_model,
    'AdaBoost': ml.ab_model,
    'Neural Network': ml.nn_model,
    'K-Neighbours': ml.kn_model
}
model = model_map.get(choice, ml.nb_model)
st.sidebar.success(f"Selected: {choice}")

# ---------------- URL input + prediction ---------------- #
st.subheader("Check a live webpage")
url = st.text_input('Full URL (http:// or https://)', placeholder='https://example.com/login')

colA, colB = st.columns([3, 1])
with colA:
    if st.button('Check now'):
        if not url or not (url.startswith('http://') or url.startswith('https://')):
            st.error('Enter a valid URL starting with http:// or https://')
        else:
            with st.spinner('Fetching page and extracting features...'):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = re.get(url, verify=False, timeout=6, headers=headers)

                    if response.status_code != 200:
                        st.error(f'Failed to fetch page, status code: {response.status_code}')
                    else:
                        soup = BeautifulSoup(response.content.decode('utf-8', errors='replace'), 'html.parser')
                        vector = fe.create_vector(soup)

                        result = model.predict([vector])

                        if show_probs and hasattr(model, 'predict_proba'):
                            probs = model.predict_proba([vector])[0]
                            st.write("### Raw Model Output (Probabilities)")
                            st.json({
                                "Legitimate (0)": float(probs[0]),
                                "Phishing (1)": float(probs[1])
                            })

                        if result[0] == 0:
                            st.success('This page looks **legitimate** ✅')
                            st.balloons()
                        else:
                            st.warning('⚠️ Potential **PHISHING** detected!')

                        # quick feature summary
                        with st.expander('Feature summary (first 10 values)'):
                            feature_names = [c for c in ml.df.columns if c != 'label']
                            preview = {feature_names[i]: vector[i] for i in range(min(10, len(vector)))}
                            st.json(preview)

                except re.exceptions.RequestException as e:
                    st.error(f'Network error: {e}')

with colB:
    st.markdown('### Examples')
    st.write('https://rtyu38.godaddysites.com/')
    st.write('https://karafuru.invite-mint.com/')
    st.write('https://defi-ned.top/h5/#/')
    st.caption('Phishing pages often have a short life — examples may expire.')

# ---------------- Footer ---------------- #
st.markdown('---')
colf1, colf2 = st.columns([3, 1])
with colf1:
    st.markdown('**About**\nEducational app for phishing detection via HTML content only. Not production-ready. Use responsibly.')
with colf2:
    st.markdown('Inspired by **Emre Kocyigit**. Streamlit UI polished by **Faria Raghib**.')

st.caption('Want extra styling (color themes, charts) or a Flask/React version? 🚀')
