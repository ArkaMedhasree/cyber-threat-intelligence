import os
import pandas as pd
import plotly.express as px
import streamlit as st

import threat_agent as ta

os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "True")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

st.set_page_config(page_title="Threat Intelligence Agent", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1300px;}
.hero {padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg,#0b1f3a,#1d4e89); color: white;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {opacity: .9; margin: 0;}
</style>
<div class="hero"><h1>🛡️ Cyber Threat Intelligence Agent</h1>
<p>Classify attacks, score severity, prioritise incidents and get remediation guidance.</p></div>
""", unsafe_allow_html=True)

SEV_COLORS = {"Low": "#2e9e5b", "Medium": "#f0b429", "High": "#e8710a", "Critical": "#c0182b"}


@st.cache_resource
def get_bundle():
    return ta.load_bundle()


@st.cache_data
def load_default(path):
    return ta.clean_logs(ta.load_logs(path))


bundle = get_bundle()

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload daily security logs (CSV)", type="csv")
default_path = os.getenv("LOG_PATH", "cyber_threat_logs.csv")
try:
    df = ta.clean_logs(ta.load_logs(uploaded)) if uploaded else load_default(default_path)
except Exception as e:
    st.error(f"Could not load logs: {e}")
    st.stop()

scored = ta.score_logs(df, bundle)

st.sidebar.header("Filters")
d_min, d_max = scored["timestamp"].min().date(), scored["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
regions = st.sidebar.multiselect("Region", sorted(scored["region"].unique()))
sevs = st.sidebar.multiselect("Predicted severity", ta.SEVERITY_ORDER)

f = scored.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    f = f[(f["timestamp"].dt.date >= date_range[0]) & (f["timestamp"].dt.date <= date_range[1])]
if regions:
    f = f[f["region"].isin(regions)]
if sevs:
    f = f[f["pred_severity"].isin(sevs)]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🚨 Daily triage", "🔎 Incident analyzer", "💬 Ask the agent"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total events", f"{len(f):,}")
    c2.metric("Critical / High", f"{f['pred_severity'].isin(['Critical','High']).sum():,}")
    c3.metric("Unresolved", f"{(f['resolved_flag'] == 0).sum():,}")
    c4.metric("Data compromised", f"{(f['data_compromised'] == 'Yes').sum():,}")
    c5.metric("Financial impact", f"${f['financial_impact_usd'].sum():,.0f}")

    left, right = st.columns(2)
    trend = f.set_index("timestamp").groupby([pd.Grouper(freq="D"), "pred_threat_type"]).size().reset_index(name="events")
    left.plotly_chart(px.line(trend, x="timestamp", y="events", color="pred_threat_type",
                              title="Attack trend over time"), width="stretch")
    sev_counts = f["pred_severity"].value_counts().reindex(ta.SEVERITY_ORDER).fillna(0).reset_index()
    sev_counts.columns = ["severity", "events"]
    right.plotly_chart(px.bar(sev_counts, x="severity", y="events", color="severity",
                              color_discrete_map=SEV_COLORS, title="Severity distribution"), width="stretch")

    l2, r2 = st.columns(2)
    top_ip = f["source_ip"].value_counts().head(10).reset_index()
    top_ip.columns = ["source_ip", "events"]
    l2.plotly_chart(px.bar(top_ip, x="events", y="source_ip", orientation="h",
                           title="Top attacking source IPs").update_yaxes(autorange="reversed"),
                    width="stretch")
    r2.plotly_chart(px.pie(f, names="pred_threat_type", hole=.45, title="Threat type mix"),
                    width="stretch")

    l3, r3 = st.columns(2)
    heat = f.groupby(["region", "pred_threat_type"]).size().reset_index(name="events")
    l3.plotly_chart(px.density_heatmap(heat, x="pred_threat_type", y="region", z="events",
                                       title="Region x threat type", color_continuous_scale="Reds"),
                    width="stretch")
    vec = f.groupby(["attack_vector", "pred_severity"]).size().reset_index(name="events")
    r3.plotly_chart(px.bar(vec, x="attack_vector", y="events", color="pred_severity",
                           color_discrete_map=SEV_COLORS, title="Attack vectors by severity"),
                    width="stretch")

    with st.expander("Model performance (hold-out set)"):
        st.json(bundle["metrics"])

with tab2:
    n = st.slider("Number of incidents to show", 5, 50, 15)
    top = ta.triage(f, n)
    st.dataframe(top, width="stretch", hide_index=True)
    st.download_button("Download triage list (CSV)", top.to_csv(index=False), "triage.csv", "text/csv")
    st.caption("Priority = 40% threat score + 40% model-expected severity + 10% data compromise + 10% anomaly score.")

with tab3:
    ids = f.sort_values("priority_score", ascending=False)["log_id"].tolist()
    if not ids:
        st.info("No incidents match the filters.")
    else:
        choice = st.selectbox("Select incident (sorted by priority)", ids[:500])
        row = f[f["log_id"] == choice].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Predicted type", row["pred_threat_type"], f"{row['type_confidence']:.0%} conf.")
        b.metric("Predicted severity", row["pred_severity"])
        c.metric("Priority", f"{row['priority_score']:.0f}/100")
        d.metric("Anomaly", f"{row['anomaly_score']:.2f}")
        if st.button("Generate AI incident summary", type="primary"):
            with st.spinner("Analyzing incident..."):
                text = ta.summarize_incident(ta.incident_context(row, df))
            with st.container(border=True):
                st.markdown(text)
        with st.expander("Raw log record"):
            st.json({k: str(v) for k, v in row.items()})

with tab4:
    st.write("Try: *top 5 threats*, *analyze log LOG00123*, *what are the main risks this week?*")
    q = st.text_input("Ask the threat intelligence agent")
    if q:
        with st.spinner("Thinking..."):
            st.markdown(ta.threat_agent(q, f, df))

