import os
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from imblearn.under_sampling import RandomUnderSampler
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# ─── Hugging Face model download & load ────────────────────────────────────────
HF_REPO     = "Abdotageldin/diabetes-pkl"
HF_FILENAME = "model_compressed.pkl"

try:
    # Download (and cache) the .pkl from your HF repo
    MODEL_PATH = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    print(f"✅ Downloaded model to '{MODEL_PATH}'")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"❌ Could not fetch/load model: {e}")

# ─── Load CSV for visualizations ───────────────────────────────────────────────
df_plot = pd.read_csv("data.csv")

# ─── Prediction form fields & feature order ───────────────────────────────────
fields = [
    {"name":"HighBP","label":"High Blood Pressure","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"HighChol","label":"High Cholesterol","type":"radio",
        "options":[{"value":0,"label":"Normal"},{"value":1,"label":"High"}]},
    {"name":"CholCheck","label":"Cholesterol Checked","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"BMI","label":"Body Mass Index (BMI)","type":"number","step":"0.1","default":25},
    {"name":"Smoker","label":"Smoker","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"Stroke","label":"Stroke History","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"HeartDiseaseorAttack","label":"Heart Disease History","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"PhysActivity","label":"Physical Activity","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"Fruits","label":"Consumes Fruits","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"Veggies","label":"Consumes Vegetables","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"HvyAlcoholConsump","label":"Heavy Alcohol Use","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"AnyHealthcare","label":"Has Healthcare Coverage","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"NoDocbcCost","label":"Skipped Doctor—Cost","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"GenHlth","label":"General Health","type":"select","options":[
       {"value":1,"label":"Excellent"},{"value":2,"label":"Very good"},
       {"value":3,"label":"Good"},{"value":4,"label":"Fair"},{"value":5,"label":"Poor"}
    ]},
    {"name":"MentHlth","label":"Poor Mental Health Days (0–30)","type":"range","min":0,"max":30,"default":0},
    {"name":"PhysHlth","label":"Poor Physical Health Days (0–30)","type":"range","min":0,"max":30,"default":0},
    {"name":"DiffWalk","label":"Difficulty Walking","type":"radio",
        "options":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}]},
    {"name":"Sex","label":"Gender","type":"radio",
        "options":[{"value":0,"label":"Female"},{"value":1,"label":"Male"}]},
    {"name":"Age","label":"Age Category","type":"select","options":[
       {"value":1,"label":"18–24 years"},{"value":2,"label":"25–29 years"},
       {"value":3,"label":"30–34 years"},{"value":4,"label":"35–39 years"},
       {"value":5,"label":"40–44 years"},{"value":6,"label":"45–49 years"},
       {"value":7,"label":"50–54 years"},{"value":8,"label":"55–59 years"},
       {"value":9,"label":"60–64 years"},{"value":10,"label":"65–69 years"},
       {"value":11,"label":"70–74 years"},{"value":12,"label":"75–79 years"},
       {"value":13,"label":"80 years and older"}
    ]},
    {"name":"Education","label":"Education Level","type":"select","options":[
       {"value":1,"label":"Never attended school or only kindergarten"},
       {"value":2,"label":"Grades 1–8 (Elementary)"},
       {"value":3,"label":"Grades 9–11 (Some high school)"},
       {"value":4,"label":"Grade 12 or GED (High school graduate)"},
       {"value":5,"label":"College 1–3 years (Some college)"},
       {"value":6,"label":"College 4+ years (College graduate)"}
    ]},
    {"name":"Income","label":"Annual Household Income","type":"select","options":[
       {"value":1,"label":"Less than $10,000"},
       {"value":2,"label":"$10,000 to < $15,000"},
       {"value":3,"label":"$15,000 to < $20,000"},
       {"value":4,"label":"$20,000 to < $25,000"},
       {"value":5,"label":"$25,000 to < $35,000"},
       {"value":6,"label":"$35,000 to < $50,000"},
       {"value":7,"label":"$50,000 to < $75,000"},
       {"value":8,"label":"$75,000 or more"}
    ]}
]
FEATURE_ORDER = [f["name"] for f in fields]

# ─── Schema for the visualizations table ────────────────────────────────────────
schema = [
    {"name":"Diabetes_012","dtype":"Categorical","description":"Diabetes status (0=No,1=Pre,2=Diabetes).","potential":"Target"},
    {"name":"HighBP","dtype":"Binary","description":"High blood pressure (0=No,1=Yes).","potential":"Comorbidity"},
    {"name":"HighChol","dtype":"Binary","description":"High cholesterol (0=No,1=Yes).","potential":"Risk factor"},
    {"name":"CholCheck","dtype":"Binary","description":"Cholesterol checked (0=No,1=Yes).","potential":"Preventative care"},
    {"name":"BMI","dtype":"Numeric","description":"Body Mass Index.","potential":"Obesity correlation"},
    {"name":"Smoker","dtype":"Binary","description":"Smoker (0=No,1=Yes).","potential":"Lifestyle risk"},
    {"name":"Stroke","dtype":"Binary","description":"Stroke history (0=No,1=Yes).","potential":"Comorbidity"},
    {"name":"HeartDiseaseorAttack","dtype":"Binary","description":"Heart disease history (0=No,1=Yes).","potential":"Comorbidity"},
    {"name":"PhysActivity","dtype":"Binary","description":"Physical activity (0=No,1=Yes).","potential":"Lifestyle factor"},
    {"name":"Fruits","dtype":"Binary","description":"Fruit intake (0=No,1=Yes).","potential":"Dietary factor"},
    {"name":"Veggies","dtype":"Binary","description":"Vegetable intake (0=No,1=Yes).","potential":"Dietary factor"},
    {"name":"HvyAlcoholConsump","dtype":"Binary","description":"Heavy alcohol use (0=No,1=Yes).","potential":"Lifestyle factor"},
    {"name":"AnyHealthcare","dtype":"Binary","description":"Has healthcare coverage (0=No,1=Yes).","potential":"Access to care"},
    {"name":"NoDocbcCost","dtype":"Binary","description":"Skipped doctor due to cost (0=No,1=Yes).","potential":"Economic barrier"},
    {"name":"GenHlth","dtype":"Categorical","description":"General health (1=Excellent…5=Poor).","potential":"Self-report"},
    {"name":"MentHlth","dtype":"Numeric","description":"Days poor mental health (0–30).","potential":"Psychological"},
    {"name":"PhysHlth","dtype":"Numeric","description":"Days poor physical health (0–30).","potential":"Physical"},
    {"name":"DiffWalk","dtype":"Binary","description":"Difficulty walking (0=No,1=Yes).","potential":"Mobility"},
    {"name":"Sex","dtype":"Binary","description":"Gender (0=Female,1=Male).","potential":"Demographic"},
    {"name":"Age","dtype":"Categorical","description":"Age group (1–13).","potential":"Demographic"},
    {"name":"Education","dtype":"Categorical","description":"Education level (1–6).","potential":"Socioeconomic"},
    {"name":"Income","dtype":"Categorical","description":"Income category (1–8).","potential":"Socioeconomic"}
]

def get_balanced_df():
    df = df_plot.copy()
    X = df.drop(columns=['Diabetes_012'])
    y = df['Diabetes_012']
    rus = RandomUnderSampler(sampling_strategy='all', random_state=27)
    X_us, y_us = rus.fit_resample(X, y)
    df_bal = X_us.copy()
    df_bal['Diabetes_012'] = y_us
    return df_bal

# ─── Main pages ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", fields=fields)

@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html", schema=schema)

@app.route("/research")
def research():
    return render_template("research.html")

# ─── API endpoints ────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify(prediction_text="❌ Model not loaded"), 500

    data = {}
    for f in fields:
        raw = request.form.get(f["name"])
        if raw is None:
            return jsonify(prediction_text=f"❌ Missing field {f['name']}"), 400
        data[f["name"]] = float(raw) if f["type"] in ("number","range","select") else int(raw)

    df = pd.DataFrame([data], columns=FEATURE_ORDER)
    label_map = {'0':'Non-Diabetic','1':'Pre-Diabetic','2':'Diabetic'}
    try:
        pred = model.predict(df)[0]
        return jsonify(prediction_text=f"⚡ Prediction: {label_map[str(int(pred))]}")
    except Exception as e:
        return jsonify(prediction_text=f"❌ Model error: {e}"), 500

@app.route("/api/visualizations", methods=["POST"])
def api_visualizations():
    col = request.form["column"]
    series = df_plot[col].dropna()
    if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 10:
        counts = series.value_counts().sort_index().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Distribution of {col}", template="plotly_dark")
        fig.update_xaxes(type="category")
    else:
        fig = px.histogram(df_plot, x=col, nbins=15, title=f"Distribution of {col}", template="plotly_dark")
    fig.update_layout(
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#121212",
        font_color="#e0e0e0",
        margin=dict(t=50,b=40,l=40,r=40),
        width=1000, height=450
    )
    return jsonify(plot_json=fig.to_json())

@app.route("/api/research/q2")
def api_q2():
    df_bal = get_balanced_df()
    cmap = {0:'#1f77b4',1:'#ff7f0e',2:'#d62728'}
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Smoking vs Diabetes","Heavy Alcohol vs Diabetes",
                        "Fruit Intake vs Diabetes","Veggies vs Diabetes"),
        vertical_spacing=0.20, horizontal_spacing=0.12
    )
    def bar(col, r, c, labels):
        grp = df_bal.groupby([col,'Diabetes_012']).size().unstack(fill_value=0)
        for s in grp.columns:
            fig.add_trace(go.Bar(
                x=grp.index, y=grp[s],
                marker_color=cmap[s],
                name=f"D{s}", text=grp[s], textposition='auto'
            ), row=r, col=c)
        fig.update_xaxes(title_text=labels, row=r, col=c)
        fig.update_yaxes(title_text="Count", row=r, col=c)

    bar('Smoker',1,1,'No/Yes')
    bar('HvyAlcoholConsump',1,2,'No/Yes')
    bar('Fruits',2,1,'<1/day,≥1/day')
    bar('Veggies',2,2,'<1/day,≥1/day')

    fig.update_layout(
        title="Lifestyle Factors vs Diabetes (Balanced)",
        template='plotly_dark', barmode='group',
        height=1000, width=1400
    )
    return jsonify(plot_json=fig.to_json())

@app.route("/api/research/q3")
def api_q3():
    df_bal = get_balanced_df()
    cmap = {0:'#1f77b4',1:'#ff7f0e',2:'#d62728'}
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Income vs Diabetes","Education vs Diabetes"),
        horizontal_spacing=0.12
    )
    income_lbl = ['<10k','10-15k','15-20k','20-25k','25-35k','35-50k','50-75k','≥75k']
    edu_lbl    = ['None/K','1-8','9-11','12/GED','Some Coll','College Grad']

    def bar(col, r, c, lbls):
        grp = df_bal.groupby([col,'Diabetes_012']).size().unstack(fill_value=0)
        for s in grp.columns:
            fig.add_trace(go.Bar(
                x=grp.index, y=grp[s],
                marker_color=cmap[s],
                name=f"D{s}", text=grp[s], textposition='auto'
            ), row=r, col=c)
        fig.update_xaxes(title_text=col, tickvals=grp.index, ticktext=lbls, row=r, col=c)
        fig.update_yaxes(title_text="Count", row=r, col=c)

    bar('Income',1,1,income_lbl)
    bar('Education',1,2,edu_lbl)

    fig.update_layout(
        title="Socioeconomic Factors vs Diabetes (Balanced)",
        template='plotly_dark', barmode='group',
        height=600, width=1400
    )
    return jsonify(plot_json=fig.to_json())

@app.route("/api/research/q4")
def api_q4():
    df_bal = get_balanced_df()
    cmap = {0:'#1f77b4',1:'#ff7f0e',2:'#d62728'}

    # Counts bar
    grp = df_bal.groupby(['PhysActivity','Diabetes_012']).size().unstack(fill_value=0)
    bar_fig = go.Figure()
    for s in grp.columns:
        bar_fig.add_trace(go.Bar(
            x=['No','Yes'], y=grp[s].values,
            marker_color=cmap[s],
            name=f"D{s}", text=grp[s].values, textposition='auto'
        ))
    bar_fig.update_layout(
        title="Physical Activity vs Diabetes (Balanced)",
        template='plotly_dark', barmode='group',
        height=500, width=800
    )

    # BMI boxplot
    box_fig = px.box(
        df_bal, x='PhysActivity', y='BMI', color='Diabetes_012',
        color_discrete_map=cmap,
        labels={'PhysActivity':'Active?','BMI':'BMI'},
        title="BMI Distribution by PhysActivity & Diabetes",
        template='plotly_dark', height=500, width=800
    )

    # Combine
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Counts","BMI Boxplot"))
    for tr in bar_fig.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in box_fig.data:
        fig.add_trace(tr, row=1, col=2)

    fig.update_layout(template='plotly_dark', barmode='group', height=550, width=1400)
    return jsonify(plot_json=fig.to_json())

@app.route("/api/research/q5")
def api_q5():
    df_bal = get_balanced_df()
    cmap = {0:'#1f77b4',1:'#ff7f0e',2:'#d62728'}
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Heart Disease vs Diabetes","Stroke vs Diabetes"),
        horizontal_spacing=0.12
    )

    def bar(col, r, c):
        grp = df_bal.groupby([col,'Diabetes_012']).size().unstack(fill_value=0)
        for s in grp.columns:
            fig.add_trace(go.Bar(
                x=['No','Yes'], y=grp[s].values,
                marker_color=cmap[s],
                name=f"D{s}", text=grp[s].values, textposition='auto'
            ), row=r, col=c)
        fig.update_yaxes(title_text="Count", row=r, col=c)

    bar('HeartDiseaseorAttack',1,1)
    bar('Stroke',1,2)

    fig.update_layout(
        title="Diabetes vs Other Conditions (Balanced)",
        template='plotly_dark', barmode='group',
        height=600, width=1400
    )
    return jsonify(plot_json=fig.to_json())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
