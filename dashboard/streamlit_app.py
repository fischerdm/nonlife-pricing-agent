"""
Streamlit dashboard for the Non-Life Pricing Agent — read-only session
viewer plus the interactive Feature & Grouping Workbench.

Run:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from dashboard import feature_workbench

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
SESSIONS_DIR = BASE_DIR / "reports" / "sessions"


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_project_config() -> dict:
    with open(CONFIG_DIR / "project_config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=60)
def load_glm_config() -> dict:
    path = CONFIG_DIR / "glm_config.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=60)
def load_all_events() -> list[dict]:
    events: list[dict] = []
    for path in sorted(SESSIONS_DIR.glob("session_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    e["_session"] = path.stem
                    events.append(e)
    return events


def last_event(events: list[dict], event_type: str) -> dict | None:
    result = None
    for e in events:
        if e["event"] == event_type:
            result = e
    return result


# ── HELPERS ───────────────────────────────────────────────────────────────────

def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def parse_patsy_param(param: str) -> tuple[str, str, bool]:
    """Return (feature, level, is_interaction) from a patsy parameter string."""
    if param == "Intercept":
        return "Intercept", "", False

    depth, top_colons = 0, []
    for i, ch in enumerate(param):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ":" and depth == 0:
            top_colons.append(i)

    if top_colons:
        return param, "", True

    m = re.match(r"^(\w+)\[T\.(.+)\]$", param)
    if m:
        return m.group(1), m.group(2), False

    return param, "", False


# ── PAGE SETUP ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Non-Life Pricing — GLM Distillation",
    page_icon="📊",
    layout="wide",
)

# ── DATA ──────────────────────────────────────────────────────────────────────

cfg = load_project_config()
glm_cfg = load_glm_config()
events = load_all_events()

numeric_features = cfg.get("features", {}).get("numeric", [])
cat_features = cfg.get("features", {}).get("categorical", [])
approved_numeric = [f for f in numeric_features if f.get("approved")]
approved_cat = [f for f in cat_features if f.get("approved")]

glm_terms_all = glm_cfg.get("glm", {}).get("terms", []) if glm_cfg else []
approved_terms_main = [t for t in glm_terms_all if t.get("approved") and t.get("term_type") == "main"]
approved_terms_inter = [t for t in glm_terms_all if t.get("approved") and t.get("term_type") == "interaction"]

rating_ev = last_event(events, "rating_factors")
gbm_ev = last_event(events, "gbm_complete")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Pricing Agent")
    st.caption("Non-Life Motor — GLM Distillation Dashboard")
    st.divider()

    feat_done = any(e["event"] == "feature_selection_complete" for e in events)
    group_done = any(e["event"] == "grouping_complete" for e in events)
    gbm_done_flag = any(e["event"] == "gbm_complete" for e in events)
    distill_done = any(e["event"] == "glm_distillation_complete" for e in events)
    glm_done = any(e["event"] == "rating_factors" for e in events)

    st.markdown("**Pipeline Stages**")
    for label, done in [
        ("Feature Selection", feat_done),
        ("Categorical Grouping", group_done),
        ("GBM Training", gbm_done_flag),
        ("GLM Distillation", distill_done),
        ("GLM Fitting", glm_done),
    ]:
        st.markdown(f"{'✅' if done else '⬜'} {label}")

    st.divider()

    sess_start = last_event(events, "session_start")
    if sess_start:
        c = sess_start["config"]
        st.markdown("**Configuration**")
        st.markdown(f"- **Target:** `{c['target_col']}`")
        st.markdown(f"- **Exposure:** `{c['exposure_col']}`")
        st.markdown(f"- **Family:** {c['objective'].title()}")
        st.markdown(f"- **LLM:** `{c['model']}`")

    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── MAIN TABS ─────────────────────────────────────────────────────────────────

(tab_overview, tab_workbench, tab_gbm, tab_glm, tab_audit) = st.tabs([
    "Overview",
    "Feature & Grouping Workbench",
    "GBM",
    "GLM",
    "Audit Trail",
])


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Pipeline Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric Features", len(approved_numeric), f"of {len(numeric_features)}")
    c2.metric("Categorical Features", len(approved_cat), f"of {len(cat_features)}")
    c3.metric("GLM Main Effects", len(approved_terms_main))
    c4.metric("GLM Interactions", len(approved_terms_inter))

    if rating_ev:
        c5, c6, c7 = st.columns(3)
        c5.metric("Deviance Explained", f"{rating_ev['deviance_explained']:.1%}")
        c6.metric("AIC", f"{rating_ev['aic']:,.0f}")
        c7.metric("Rating Parameters", len(rating_ev.get("rating_factors", [])))

    st.divider()

    col_n, col_c = st.columns(2)
    with col_n:
        st.markdown("**Approved Numeric Features**")
        for f in approved_numeric:
            st.markdown(f"- `{f['name']}`")
    with col_c:
        st.markdown("**Approved Categorical Features**")
        for f in approved_cat:
            n_groups = len(f.get("grouping") or {}) or f.get("n_clusters", "?")
            st.markdown(f"- `{f['name']}` — {n_groups} groups")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE & GROUPING WORKBENCH
# ══════════════════════════════════════════════════════════════════════════════

with tab_workbench:
    st.header("Feature & Grouping Workbench")
    st.caption(
        "The agent proposes a feature list and groups every categorical in one combined "
        "draft. Review each variable below — include or exclude it, leave a comment — "
        "then re-run the agent with your feedback. Repeat until you finalize the selection; "
        "finalizing writes the checkpoint that the GBM and GLM stages read from."
    )
    feature_workbench.render_feature_workbench(cfg, CONFIG_DIR / "project_config.yaml")


# ══════════════════════════════════════════════════════════════════════════════
# GBM
# ══════════════════════════════════════════════════════════════════════════════

with tab_gbm:
    st.header("GBM — LightGBM Feature Analysis")

    if not gbm_ev:
        st.info("GBM data not yet available.")
    else:
        st.subheader("Feature Importance (Gain)")

        fi = gbm_ev.get("feature_importances", [])
        df_fi = pd.DataFrame(fi).sort_values("importance")
        df_fi["Importance (%)"] = (df_fi["importance"] * 100).round(2)
        df_fi = df_fi.rename(columns={"feature": "Feature"})

        fig_fi = px.bar(
            df_fi,
            x="Importance (%)",
            y="Feature",
            orientation="h",
            color="Importance (%)",
            color_continuous_scale="Blues",
        )
        fig_fi.update_coloraxes(showscale=False)
        fig_fi.update_layout(height=430, margin=dict(l=0, r=20, t=10, b=0))
        st.plotly_chart(fig_fi, use_container_width=True)

        st.divider()
        st.subheader("Pairwise Interactions (H-Statistics)")
        st.caption(
            "Friedman H-statistic measures the fraction of variance explained by the interaction "
            "of two features. Higher = stronger interaction. Only non-zero pairs shown."
        )

        interactions = gbm_ev.get("interactions", []) or cfg.get("gbm_output", {}).get("interactions", [])
        non_zero = [i for i in interactions if i["h_statistic"] > 0]

        c_slider, _ = st.columns([1, 3])
        n_top = c_slider.slider("Top N interactions", 5, min(50, len(non_zero)), 20)

        df_h = pd.DataFrame(non_zero[:n_top]).copy()
        df_h["Pair"] = df_h["feature_a"] + " × " + df_h["feature_b"]
        df_h = df_h.sort_values("h_statistic")

        fig_h = px.bar(
            df_h,
            x="h_statistic",
            y="Pair",
            orientation="h",
            color="h_statistic",
            color_continuous_scale="Oranges",
            labels={"h_statistic": "H-Statistic", "Pair": ""},
        )
        fig_h.update_coloraxes(showscale=False)
        fig_h.update_layout(height=max(350, n_top * 26), margin=dict(l=0, r=20, t=10, b=0))
        st.plotly_chart(fig_h, use_container_width=True)

        with st.expander("Full H-statistic table"):
            df_all = pd.DataFrame(interactions)
            df_all["Pair"] = df_all["feature_a"] + " × " + df_all["feature_b"]
            st.dataframe(
                df_all[["Pair", "feature_a", "feature_b", "h_statistic"]].rename(columns={
                    "feature_a": "Feature A",
                    "feature_b": "Feature B",
                    "h_statistic": "H-Statistic",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# GLM
# ══════════════════════════════════════════════════════════════════════════════

with tab_glm:
    st.header("GLM — Gamma Log-Link")

    if rating_ev:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deviance Explained", f"{rating_ev['deviance_explained']:.2%}")
        c2.metric("AIC", f"{rating_ev['aic']:,.0f}")
        c3.metric("Main Effects", len(approved_terms_main))
        c4.metric("Interactions", len(approved_terms_inter))

    formula = glm_cfg.get("glm", {}).get("formula", "") if glm_cfg else ""
    if formula:
        with st.expander("Model formula"):
            st.code(formula, language=None)

    glm_sub = st.tabs(["Main Effects", "Interactions", "Rating Factors", "Relativity Chart"])

    # ── Main Effects ──────────────────────────────────────────────────────────
    with glm_sub[0]:
        rows = []
        for t in approved_terms_main:
            rows.append({
                "Feature": t["name"],
                "Rationale": t.get("rationale", ""),
                "Actuary Note": t.get("actuary_note", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Interactions ──────────────────────────────────────────────────────────
    with glm_sub[1]:
        rows = []
        for t in approved_terms_inter:
            rows.append({
                "Interaction Term": t["name"],
                "H-Statistic": round(t.get("h_statistic") or 0, 4),
                "Rationale": t.get("rationale", ""),
                "Actuary Note": t.get("actuary_note", ""),
            })
        df_inter = pd.DataFrame(rows)
        if not df_inter.empty:
            df_inter = df_inter.sort_values("H-Statistic", ascending=False)
        st.dataframe(df_inter, use_container_width=True, hide_index=True)

    # ── Rating Factors ────────────────────────────────────────────────────────
    with glm_sub[2]:
        if not rating_ev:
            st.info("GLM not yet fitted.")
        else:
            coefs = rating_ev.get("rating_factors", [])

            rf_rows = []
            for c in coefs:
                param = c["parameter"]
                feat, level, is_inter = parse_patsy_param(param)
                rf_rows.append({
                    "Parameter": param,
                    "Feature": feat,
                    "Level": level,
                    "Type": "interaction" if is_inter else ("intercept" if feat == "Intercept" else "main"),
                    "Relativity": round(c["exp_coef"], 4),
                    "Sig.": sig_stars(c["p_value"]),
                    "p-value": round(c["p_value"], 6),
                    "CI Lower": round(c.get("ci_lower_exp", float("nan")), 4),
                    "CI Upper": round(c.get("ci_upper_exp", float("nan")), 4),
                    "log(coef)": round(c["coef"], 4),
                })

            df_rf = pd.DataFrame(rf_rows)

            fc1, fc2, fc3 = st.columns(3)
            search = fc1.text_input("Filter parameter", placeholder="e.g. bonus_score")
            type_sel = fc2.selectbox("Type", ["All", "main", "interaction", "intercept"])
            sig_only = fc3.checkbox("Significant only (p < 0.05)")

            df_show = df_rf.copy()
            if search:
                mask = df_show["Parameter"].str.contains(search, case=False, na=False)
                df_show = df_show[mask]
            if type_sel != "All":
                df_show = df_show[df_show["Type"] == type_sel]
            if sig_only:
                df_show = df_show[df_show["p-value"] < 0.05]

            st.caption(f"{len(df_show)} of {len(df_rf)} parameters")

            def color_relativity(val: float) -> str:
                if val > 1.3:
                    return "background-color: #ffcccc"
                if val > 1.1:
                    return "background-color: #ffe0cc"
                if val < 0.7:
                    return "background-color: #cce0ff"
                if val < 0.9:
                    return "background-color: #e3f2fd"
                return ""

            st.dataframe(
                df_show.style.map(color_relativity, subset=["Relativity"]),
                column_config={
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                    "Relativity": st.column_config.NumberColumn(format="%.4f"),
                    "CI Lower": st.column_config.NumberColumn(format="%.4f"),
                    "CI Upper": st.column_config.NumberColumn(format="%.4f"),
                },
                use_container_width=True,
                hide_index=True,
            )

    # ── Relativity Chart ──────────────────────────────────────────────────────
    with glm_sub[3]:
        if not rating_ev:
            st.info("GLM not yet fitted.")
        else:
            coefs = rating_ev.get("rating_factors", [])

            chart_rows = []
            for c in coefs:
                param = c["parameter"]
                feat, level, is_inter = parse_patsy_param(param)
                if feat == "Intercept" or is_inter:
                    continue
                chart_rows.append({
                    "Parameter": param,
                    "Feature": feat,
                    "Level": level if level else feat,
                    "Relativity": c["exp_coef"],
                    "CI Lower": c.get("ci_lower_exp", c["exp_coef"]),
                    "CI Upper": c.get("ci_upper_exp", c["exp_coef"]),
                    "Significant": c["p_value"] < 0.05,
                })

            df_chart = pd.DataFrame(chart_rows)

            if df_chart.empty:
                st.info("No main effect parameters to chart.")
            else:
                all_feats = sorted(df_chart["Feature"].unique())
                sel = st.multiselect(
                    "Select features to display",
                    all_feats,
                    default=all_feats[:6],
                )
                df_plot = df_chart[df_chart["Feature"].isin(sel)] if sel else df_chart

                fig = px.scatter(
                    df_plot,
                    x="Level",
                    y="Relativity",
                    color="Feature",
                    error_y=df_plot["CI Upper"] - df_plot["Relativity"],
                    error_y_minus=df_plot["Relativity"] - df_plot["CI Lower"],
                    symbol="Significant",
                    symbol_map={True: "circle", False: "x"},
                    labels={"Level": "Parameter", "Relativity": "Relativity (exp coef)"},
                    height=520,
                )
                fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="base = 1.0",
                    annotation_position="right",
                )
                fig.update_layout(margin=dict(l=0, r=80, t=30, b=0))
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Solid circles = statistically significant (p < 0.05). "
                    "X marks = not significant. Error bars = 95% CI."
                )


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════

with tab_audit:
    st.header("Actuary Decision Audit Trail")

    DECISION_EVENTS = {"feature_decision", "grouping_decision", "glm_term_decision", "glm_coef_decision"}
    ICONS: dict[str, str] = {
        "approved": "✅", "rejected": "❌", "noted": "📝",
        "skipped": "⏭️", "kept": "✅", "quit": "🚪",
    }

    audit_rows = []
    for e in events:
        if e["event"] not in DECISION_EVENTS:
            continue

        evt = e["event"]

        if evt == "feature_decision":
            item = e.get("feature", "")
            decision = e.get("decision", "")
            note = e.get("note", "")
        elif evt == "grouping_decision":
            item = f"{e.get('col_name', '')} → {e.get('cluster', '')}"
            decision = e.get("decision", "")
            note = e.get("note", "")
        elif evt == "glm_term_decision":
            item = e.get("term", "")
            decision = e.get("decision", "")
            note = e.get("note", "")
        elif evt == "glm_coef_decision":
            item = e.get("term", "")
            decision = e.get("decision", "")
            note = ""
        else:
            continue

        audit_rows.append({
            "Timestamp": e["ts"][:19].replace("T", " "),
            "Stage": e.get("stage", ""),
            "Item": item,
            "Decision": f"{ICONS.get(decision, '')} {decision}",
            "Note": note or "",
        })

    if not audit_rows:
        st.info("No decision events found in session logs.")
    else:
        df_audit = pd.DataFrame(audit_rows)

        fc1, fc2 = st.columns(2)
        stage_opts = ["All"] + sorted(df_audit["Stage"].unique().tolist())
        stage_filter = fc1.selectbox("Stage", stage_opts)
        notes_only = fc2.checkbox("Only show decisions with notes")

        df_show = df_audit.copy()
        if stage_filter != "All":
            df_show = df_show[df_show["Stage"] == stage_filter]
        if notes_only:
            df_show = df_show[df_show["Note"] != ""]

        st.caption(f"{len(df_show)} decisions")
        st.dataframe(df_show, use_container_width=True, hide_index=True)
