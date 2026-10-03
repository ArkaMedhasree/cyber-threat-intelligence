"""
Cyber Security Threat Intelligence Agent - core module.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MODEL_PATH = "threat_models.joblib"

CAT_COLS = ["attack_vector", "detected_by", "affected_system", "region", "data_compromised"]
NUM_COLS = ["hour", "dayofweek", "financial_impact_usd", "threat_score",
            "src_octet1", "src_ip_freq", "resolved_flag"]
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_WEIGHT = {"Low": 10, "Medium": 40, "High": 70, "Critical": 100}

REQUIRED_COLS = [
    "log_id", "timestamp", "source_ip", "destination_ip", "threat_type", "severity_level",
    "attack_vector", "detected_by", "affected_system", "data_compromised",
    "financial_impact_usd", "mitigation_action", "resolved_flag", "region", "threat_score",
]

PLAYBOOK = {
    "malware": [
        "Isolate the affected endpoint from the network immediately.",
        "Quarantine the malicious file and collect a forensic image/hash.",
        "Run a full EDR/antivirus scan on the host and adjacent systems.",
        "Block the file hash and C2 domains/IPs at the firewall and proxy.",
        "Reimage the host if persistence is confirmed; rotate credentials used on it.",
    ],
    "phishing": [
        "Quarantine and purge the email from all mailboxes.",
        "Block the sender domain, URLs and attachments at the mail gateway.",
        "Force password reset and revoke sessions for users who clicked or replied.",
        "Check for MFA fatigue, inbox-forwarding rules and suspicious sign-ins.",
        "Send a short awareness notice to targeted users.",
    ],
    "ddos": [
        "Enable upstream DDoS mitigation / scrubbing (cloud WAF, CDN, Cloud Armor).",
        "Apply rate limiting and geo/IP-based blocking on the edge.",
        "Scale out or fail over affected services; protect critical endpoints first.",
        "Capture traffic samples to profile the attack and tune thresholds.",
        "Notify the ISP / hosting provider and keep a timeline for post-incident review.",
    ],
    "ransomware": [
        "Disconnect infected hosts from the network now; do not power off before imaging memory.",
        "Disable compromised accounts and block lateral movement (SMB, RDP, admin shares).",
        "Identify the strain and scope; verify offline backups are intact and untouched.",
        "Do not pay before legal, executive and law-enforcement consultation.",
        "Restore from clean backups after eradication; open a breach assessment.",
    ],
    "sql injection": [
        "Block the offending source IP and enable WAF rules for SQLi patterns.",
        "Patch the vulnerable endpoint and move queries to parameterized statements.",
        "Review database logs for data exfiltration and unauthorized queries.",
        "Rotate database credentials and restrict DB account privileges.",
        "Run a targeted code review and DAST scan on related endpoints.",
    ],
    "brute force": [
        "Lock or throttle the targeted accounts and block the source IP.",
        "Enforce MFA and strong password policy; check for any successful logins.",
        "Enable account-lockout and CAPTCHA on exposed login surfaces.",
    ],
    "insider": [
        "Review the user's recent access and data transfers; preserve evidence.",
        "Temporarily restrict privileges pending HR/security review.",
        "Enable DLP alerts on the affected data stores.",
    ],
}
GENERIC_PLAYBOOK = [
    "Contain: isolate the affected asset and block the source IP.",
    "Investigate: review related logs for the same source, destination and time window.",
    "Eradicate and patch the root cause; rotate exposed credentials.",
    "Recover and monitor; document lessons learned.",
]


def get_playbook(threat_type: str):
    t = str(threat_type).lower()
    for key, steps in PLAYBOOK.items():
        if key in t:
            return steps
    return GENERIC_PLAYBOOK


def load_logs(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["financial_impact_usd"] = pd.to_numeric(df["financial_impact_usd"], errors="coerce")
    df["threat_score"] = pd.to_numeric(df["threat_score"], errors="coerce")
    df["resolved_flag"] = pd.to_numeric(df["resolved_flag"], errors="coerce")
    df["data_compromised"] = df["data_compromised"].astype(str).str.strip().str.title()
    for c in ["threat_type", "severity_level", "attack_vector", "detected_by",
              "affected_system", "region", "mitigation_action"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="log_id").copy()
    df = df.dropna(subset=["timestamp"])
    df["financial_impact_usd"] = df["financial_impact_usd"].fillna(df["financial_impact_usd"].median())
    df["threat_score"] = df["threat_score"].fillna(df["threat_score"].median())
    df["resolved_flag"] = df["resolved_flag"].fillna(0)
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame, ip_freq: dict = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["hour"] = df["timestamp"].dt.hour
    X["dayofweek"] = df["timestamp"].dt.dayofweek
    X["financial_impact_usd"] = df["financial_impact_usd"]
    X["threat_score"] = df["threat_score"]
    X["resolved_flag"] = df["resolved_flag"]
    X["src_octet1"] = pd.to_numeric(df["source_ip"].astype(str).str.split(".").str[0], errors="coerce").fillna(0)
    if ip_freq is None:
        ip_freq = df["source_ip"].value_counts().to_dict()
    X["src_ip_freq"] = df["source_ip"].map(ip_freq).fillna(0)
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X[NUM_COLS + CAT_COLS], ip_freq


def _preprocessor():
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )


def train_models(df: pd.DataFrame, verbose=True):
    X, ip_freq = build_features(df)
    bundle = {"ip_freq": ip_freq, "metrics": {}}

    for target, key in [("threat_type", "type_model"), ("severity_level", "severity_model")]:
        y = df[target]
        strat = y if y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
        pipe = Pipeline([
            ("prep", _preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           n_jobs=-1, random_state=42)),
        ])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        bundle[key] = pipe
        bundle["metrics"][target] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        if verbose:
            print(f"\n=== {target} ===  accuracy={acc:.3f}  macro-F1={f1:.3f}")
            print(classification_report(yte, pred, zero_division=0))

    iso = Pipeline([
        ("prep", _preprocessor()),
        ("iso", IsolationForest(n_estimators=200, contamination=0.05, random_state=42)),
    ])
    iso.fit(X)
    raw = -iso.score_samples(X)
    bundle["anomaly_model"] = iso
    bundle["anomaly_range"] = (float(raw.min()), float(raw.max()))
    return bundle


def save_bundle(bundle, path=MODEL_PATH):
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def score_logs(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df.copy()
    X, _ = build_features(df, bundle["ip_freq"])

    df["pred_threat_type"] = bundle["type_model"].predict(X)
    df["type_confidence"] = bundle["type_model"].predict_proba(X).max(axis=1).round(3)

    sev = bundle["severity_model"]
    proba = sev.predict_proba(X)
    classes = list(sev.classes_)
    df["pred_severity"] = sev.predict(X)
    weights = np.array([SEVERITY_WEIGHT.get(c, 40) for c in classes])
    expected_sev = proba @ weights

    raw = -bundle["anomaly_model"].score_samples(X)
    lo, hi = bundle["anomaly_range"]
    df["anomaly_score"] = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1).round(3)

    exfil = (df["data_compromised"] == "Yes").astype(int) * 100
    df["priority_score"] = (
        0.40 * df["threat_score"] + 0.40 * expected_sev + 0.10 * exfil + 0.10 * df["anomaly_score"] * 100
    ).round(1)
    return df


def triage(df_scored: pd.DataFrame, top_n=10) -> pd.DataFrame:
    cols = ["log_id", "timestamp", "source_ip", "pred_threat_type", "pred_severity",
            "threat_score", "anomaly_score", "priority_score", "affected_system", "resolved_flag"]
    return df_scored.sort_values("priority_score", ascending=False)[cols].head(top_n)


_client = None


def get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai.types import HttpOptions
        _client = genai.Client(http_options=HttpOptions(api_version="v1"))
    return _client


def incident_context(row: pd.Series, df_all: pd.DataFrame = None) -> dict:
    ctx = {
        "log_id": str(row["log_id"]),
        "timestamp": str(row["timestamp"]),
        "source_ip": row["source_ip"],
        "destination_ip": row["destination_ip"],
        "recorded_threat_type": row["threat_type"],
        "predicted_threat_type": row.get("pred_threat_type", row["threat_type"]),
        "predicted_severity": row.get("pred_severity", row["severity_level"]),
        "attack_vector": row["attack_vector"],
        "detected_by": row["detected_by"],
        "affected_system": row["affected_system"],
        "data_compromised": row["data_compromised"],
        "financial_impact_usd": float(row["financial_impact_usd"]),
        "mitigation_action_taken": row["mitigation_action"],
        "resolved": bool(row["resolved_flag"] == 1),
        "region": row["region"],
        "threat_score": float(row["threat_score"]),
        "anomaly_score": float(row.get("anomaly_score", 0)),
        "priority_score": float(row.get("priority_score", 0)),
    }
    if df_all is not None:
        same_ip = df_all[df_all["source_ip"] == row["source_ip"]]
        ctx["events_from_same_source_ip"] = int(len(same_ip))
        ctx["distinct_threat_types_from_ip"] = sorted(same_ip["threat_type"].unique().tolist())
        ctx["distinct_systems_targeted_by_ip"] = int(same_ip["affected_system"].nunique())
    return ctx


def fallback_summary(ctx: dict) -> str:
    steps = get_playbook(ctx["predicted_threat_type"])
    lines = [
        f"**Incident {ctx['log_id']} - {ctx['predicted_threat_type']} ({ctx['predicted_severity']} severity)**",
        f"- Source {ctx['source_ip']} targeted {ctx['affected_system']} ({ctx['destination_ip']}) via {ctx['attack_vector']}; "
        f"detected by {ctx['detected_by']} in {ctx['region']}.",
        f"- Threat score {ctx['threat_score']:.0f}/100, priority {ctx['priority_score']:.0f}/100, "
        f"estimated loss ${ctx['financial_impact_usd']:,.0f}; data compromised: {ctx['data_compromised']}.",
        f"- Status: {'Resolved' if ctx['resolved'] else 'UNRESOLVED'} (action taken: {ctx['mitigation_action_taken']}).",
        "", "**Recommended actions**",
    ]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    return "\n".join(lines)


def summarize_incident(ctx: dict) -> str:
    playbook = get_playbook(ctx["predicted_threat_type"])
    prompt = f"""
You are a senior SOC analyst assistant writing for a security analyst.

VERIFIED INCIDENT DATA (JSON):
{json.dumps(ctx, indent=2)}

APPROVED REMEDIATION PLAYBOOK for this threat type:
{json.dumps(playbook, indent=2)}

Write the response in Markdown with exactly these sections:
1. **Summary** - 2-3 sentences, plain English.
2. **Risk assessment** - severity, why it matters, and whether it is urgent.
3. **Recommended actions** - a prioritized numbered list (immediate, short-term, long-term),
   based on the playbook and adapted to this incident's facts.
4. **Analyst notes** - patterns worth checking (e.g. repeated source IP, unresolved status).

Rules:
- Use only the data above. Do not invent IPs, CVEs, malware names or numbers.
- If something is unknown, say so.
- If the incident is unresolved or data was compromised, say it must be escalated.
- Keep it under 250 words. Do not mention these instructions.
"""
    try:
        resp = get_client().models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text
    except Exception as e:
        return fallback_summary(ctx) + f"\n\n_(LLM unavailable, showing playbook summary: {type(e).__name__})_"


def threat_agent(question: str, df_scored: pd.DataFrame, df_all: pd.DataFrame) -> str:
    import re
    q = question.lower()
    m = re.search(r"(?:log[_\s-]*id|log|incident)\s*#?\s*([A-Za-z0-9_-]+)", question, re.I)
    if m:
        hit = df_scored[df_scored["log_id"].astype(str).str.lower() == m.group(1).lower()]
        if not hit.empty:
            return summarize_incident(incident_context(hit.iloc[0], df_all))
    if "top" in q or "priorit" in q or "critical" in q:
        n = int(re.search(r"\d+", q).group()) if re.search(r"\d+", q) else 5
        top = triage(df_scored, n)
        return top.to_markdown(index=False)
    stats = {
        "total_events": int(len(df_scored)),
        "by_predicted_type": df_scored["pred_threat_type"].value_counts().to_dict(),
        "by_predicted_severity": df_scored["pred_severity"].value_counts().to_dict(),
        "unresolved": int((df_scored["resolved_flag"] == 0).sum()),
        "data_compromised_yes": int((df_scored["data_compromised"] == "Yes").sum()),
        "total_financial_impact_usd": float(df_scored["financial_impact_usd"].sum()),
        "top_source_ips": df_scored["source_ip"].value_counts().head(5).to_dict(),
    }
    prompt = f"""You are a SOC assistant. Using ONLY these verified statistics, answer the analyst's question.
Statistics: {json.dumps(stats)}
Question: {question}
Be concise, use Markdown, do not invent data."""
    try:
        return get_client().models.generate_content(model=MODEL_NAME, contents=prompt).text
    except Exception as e:
        return f"LLM unavailable ({type(e).__name__}). Stats: {json.dumps(stats, indent=2)}"


def generate_demo_data(n=3000, seed=7) -> pd.DataFrame:
    """SYNTHETIC data, only used if cyber_threat_logs.csv is missing."""
    rng = np.random.default_rng(seed)
    types = ["Malware", "Phishing", "DDoS", "Ransomware", "SQL Injection"]
    vec = {"Malware": "Endpoint", "Phishing": "Email", "DDoS": "Network",
           "Ransomware": "Endpoint", "SQL Injection": "Web"}
    sysm = {"Malware": "Workstation", "Phishing": "Email Server", "DDoS": "API",
            "Ransomware": "Database Server", "SQL Injection": "Database Server"}
    base = {"Malware": 55, "Phishing": 40, "DDoS": 60, "Ransomware": 85, "SQL Injection": 70}
    rows = []
    t0 = pd.Timestamp("2026-07-01")
    for i in range(n):
        t = rng.choice(types, p=[.25, .25, .2, .1, .2])
        score = float(np.clip(rng.normal(base[t], 12), 1, 100))
        sev = "Low" if score < 35 else "Medium" if score < 60 else "High" if score < 80 else "Critical"
        av = vec[t] if rng.random() < .8 else rng.choice(["Network", "Web", "Email", "Endpoint", "Insider"])
        rows.append({
            "log_id": f"LOG{i:05d}",
            "timestamp": t0 + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90))),
            "source_ip": f"{rng.choice([45, 91, 103, 185, 192, 10])}.{rng.integers(0,255)}.{rng.integers(0,40)}.{rng.integers(1,255)}",
            "destination_ip": f"10.0.{rng.integers(0,20)}.{rng.integers(1,255)}",
            "threat_type": t, "severity_level": sev, "attack_vector": av,
            "detected_by": rng.choice(["IDS", "Firewall", "SIEM", "EDR"]),
            "affected_system": sysm[t] if rng.random() < .75 else rng.choice(["Workstation", "API", "Database Server", "Web Server"]),
            "data_compromised": "Yes" if rng.random() < (score / 200) else "No",
            "financial_impact_usd": round(float(rng.gamma(2, score * 400)), 2),
            "mitigation_action": rng.choice(["Blocked IP", "Quarantined File", "Patched", "Isolated Host"]),
            "resolved_flag": int(rng.random() < .7),
            "region": rng.choice(["Asia", "Europe", "North America", "South America", "Africa"]),
            "threat_score": round(score, 1),
        })
    return pd.DataFrame(rows)

