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

from core.distillation_pipeline import list_glm_draft_snapshots
from core.feature_pipeline import list_draft_snapshots
from core.gbm_pipeline import list_gbm_runs
from core.schemas import CommentEntry
from core.snapshot_utils import format_ts, snapshot_ts
from dashboard import feature_workbench, gbm_workbench, glm_coef_workbench, glm_workbench
from dashboard._comments import render_comment_history

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
SESSIONS_DIR = BASE_DIR / "reports" / "sessions"


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_project_config() -> dict:
    with open(CONFIG_DIR / "project_config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=5)
def load_glm_config() -> dict:
    path = CONFIG_DIR / "glm_config.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=5)
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


def _split_top_level_colons(param: str) -> list[str]:
    """Split a patsy parameter string on ':' at bracket depth 0 — patsy joins
    an interaction's sides with ':', but a level name itself never contains
    one at the top level, only (harmlessly) inside a "[T.xxx]" it's tracking
    depth around here for. A main effect (no top-level ':') comes back as a
    single-element list, unsplit."""
    depth, cuts = 0, []
    for i, ch in enumerate(param):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ":" and depth == 0:
            cuts.append(i)
    bounds = [-1, *cuts, len(param)]
    return [param[bounds[i] + 1:bounds[i + 1]] for i in range(len(bounds) - 1)]


def parse_patsy_param(param: str) -> tuple[str, str, bool]:
    """Return (feature, level, is_interaction) from a patsy parameter string."""
    if param == "Intercept":
        return "Intercept", "", False

    parts = _split_top_level_colons(param)
    if len(parts) > 1:
        return param, "", True

    m = re.match(r"^(\w+)\[T\.(.+)\]$", param)
    if m:
        return m.group(1), m.group(2), False

    return param, "", False


def _param_base_term(param: str) -> str:
    """The formula term a fitted coefficient's parameter string belongs to —
    e.g. both "policy_type[T.a]:business_type[T.x]" and
    "policy_type[T.b]:business_type[T.x]" -> "policy_type:business_type".
    A categorical's every level, or an interaction's every level
    combination, gets its own coefficient row but is one formula term —
    stripping each side's "[T.<level>]" dummy-coding suffix before
    rejoining with ':' collapses them back to that one term."""
    def base(part: str) -> str:
        m = re.match(r"^(\w+)\[T\.(.+)\]$", part)
        return m.group(1) if m else part

    return ":".join(base(p) for p in _split_top_level_colons(param))


def _term_counts(rating_factors: list[dict]) -> tuple[int, int]:
    """(# main-effect terms, # interaction terms) actually present in one
    fit's own `rating_factors` rows — the term set that fit was run with,
    which can differ from whatever's *currently* approved in glm_config.yaml
    if terms were added or removed after that fit ran. Deduped by base term
    name (`_param_base_term`), not counted per coefficient row — a
    9-level categorical is one main-effect term, not 8 rows' worth."""
    main_terms: set[str] = set()
    inter_terms: set[str] = set()
    for row in rating_factors:
        param = row["parameter"]
        if param == "Intercept":
            continue
        base = _param_base_term(param)
        (inter_terms if ":" in base else main_terms).add(base)
    return len(main_terms), len(inter_terms)


def _term_note_entries(term: dict) -> list[CommentEntry]:
    """Union of a GLM term's `comment_history` and any lingering `actuary_note`,
    as real `CommentEntry` objects so `render_comment_history` shows the actual
    Claude logo — same rendering as the workbenches, not a stand-in emoji.

    `comment_history` is the durable record of actual actuary/agent back-and-forth
    — empty for a term that was never remarked on through an Update round.
    `actuary_note` is meant to be purely transient (folded into history and
    cleared the moment a refine call runs), but for a term finalized straight
    from its first proposal with no refine round at all, it never gets folded —
    most commonly the distillation agent's own initial-proposal caveat on an
    interaction it flagged as borderline (see prompts/distillation.yaml's "flag
    any interaction that appears spurious"). That's why this is empty for every
    main effect by design: the prompt only asks for that caveat on interactions.
    """
    entries = [CommentEntry(**e) for e in (term.get("comment_history") or [])]
    if term.get("actuary_note"):
        entries.append(CommentEntry(author="agent", text=term["actuary_note"], ts=""))
    return entries


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _position_label(ts: str, history: list[str]) -> str:
    """Where `ts` sits in `history` (newest first) — "latest", "2nd latest",
    etc. — rather than a bare "(current)" that tells the actuary nothing they
    couldn't already infer (current always *is* the latest, by construction).
    A real position answers the question that actually matters: was this
    stage built from the newest available upstream version, or did whoever
    ran it deliberately reach back for an older one? Empty if `ts` can't be
    placed (missing, or not found in `history` — e.g. GBM has no snapshot
    file, so a run older than what's still in the session log has dropped
    out of the list it'd be looked up against)."""
    if not ts or ts not in history:
        return ""
    idx = history.index(ts)
    return "latest" if idx == 0 else f"{_ordinal(idx + 1)} latest"


def _source_stale(source: dict | None, history: list[str]) -> bool:
    """True if `source`'s recorded upstream ts is no longer the newest entry
    in `history` — i.e. the upstream stage has since finalized/trained a
    newer version than the one this stage actually ran against. `history` is
    recomputed fresh on every render (see `feature_history`/`gbm_history`/
    `distill_history` below), so this reflects the *current* state of the
    world, not just what was true when the source was recorded.

    Resolves `source`'s timestamp via `_source_ts` — same fallback
    `_fmt_built_from` uses for display, so a pre-migration source with no
    top-level `ts` (only the older `label`-only shape) still gets checked
    correctly instead of silently never showing stale. A source that's
    missing entirely, or whose ts has since dropped out of `history`
    altogether, can't be checked this way and is never flagged stale from
    this signal alone."""
    ts = _source_ts(source)
    return bool(ts) and ts in history and history.index(ts) != 0


def render_pipeline_graph(stages: list[tuple[str, bool, str]]) -> None:
    """Vertical stage graph for the sidebar: a solid, filled node + solid
    connector for a stage that's both completed *and* still current, a
    dashed, hollow node + dashed connector for one that's either still
    pending or has gone stale (a `reason` string, shown as a hover tooltip,
    distinguishes the two — "Not yet run" vs. "Stale: ..."). Deliberately one
    visual "not solid" state, not two, per the actuary's own framing: staleness
    should read the same as not-yet-run, since both mean "don't trust this
    stage's output as current." Replaces the old flat ✅/⬜ tick list with
    something that reads as one continuous pipeline rather than a checklist.

    `stages` is `(label, solid, reason)` — `solid=True` iff the stage has a
    completion event (from the same session-log flags the Audit Trail reads)
    AND every upstream stage's recorded source is still the newest available
    (see `_source_stale`), cascaded so one stale upstream propagates all the
    way downstream. This is a renderer, not a second source of truth for
    either fact — both are computed by the caller.

    Colors lean on `currentColor` (inherits Streamlit's own theme text color,
    so it's legible in light/dark/auto without hardcoding a hex) plus the
    app's configured accent for the "done" state, so a custom theme's
    primaryColor is picked up automatically instead of a hardcoded default.
    """
    accent = st.get_option("theme.primaryColor") or "#FF4B4B"
    rows = []
    n = len(stages)
    for i, (label, solid, reason) in enumerate(stages):
        node_cls = "done" if solid else "pending"
        title = f' title="{reason}"' if reason else ""
        rows.append(
            f'<div class="pg-row">'
            f'<div class="pg-rail">'
            f'<span class="pg-dot pg-{node_cls}"{title}></span>'
            + (f'<span class="pg-edge pg-{node_cls}"></span>' if i < n - 1 else "")
            + f'</div>'
            f'<span class="pg-label pg-{node_cls}"{title}>{label}</span>'
            f'</div>'
        )
    st.markdown(
        f"""
        <style>
        .pg-row {{ display: flex; align-items: flex-start; min-height: 2.1rem; }}
        .pg-rail {{ display: flex; flex-direction: column; align-items: center;
                    width: 1.1rem; flex-shrink: 0; }}
        .pg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-top: 4px;
                   box-sizing: border-box; flex-shrink: 0; }}
        .pg-dot.pg-done {{ background: {accent}; border: 2px solid {accent}; }}
        .pg-dot.pg-pending {{ background: transparent;
                               border: 2px dashed currentColor; opacity: 0.4; }}
        .pg-edge {{ width: 0; height: 1.3rem; margin-top: 2px; }}
        .pg-edge.pg-done {{ border-left: 2px solid {accent}; }}
        .pg-edge.pg-pending {{ border-left: 2px dashed currentColor; opacity: 0.4; }}
        .pg-label {{ margin-left: 0.5rem; font-size: 0.95rem; line-height: 1.6; }}
        .pg-label.pg-done {{ font-weight: 600; }}
        .pg-label.pg-pending {{ opacity: 0.55; }}
        </style>
        {"".join(rows)}
        """,
        unsafe_allow_html=True,
    )


# ── PAGE SETUP ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Non-Life Pricing — GLM Distillation",
    page_icon=":material/monitoring:",
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
distill_ev = last_event(events, "glm_distillation_complete")

# Newest-first timestamp histories for each stage, so a lineage reference can
# be placed by position ("latest", "2nd latest", ...) rather than a bare
# "(current)" that tells the actuary nothing they couldn't already infer
# (current always *is* the latest). Shared by the GLM Results "Fit History"
# table and the Audit Trail's "Model Lineage" table — one implementation.
feature_history = [snapshot_ts(p, "feature_draft_") for p in list_draft_snapshots("finalized")]
gbm_history = [format_ts(r.get("ts")) for r in list_gbm_runs()]
distill_history = [snapshot_ts(p, "glm_draft_") for p in list_glm_draft_snapshots("finalized")]


def _fmt_ts(e_or_ts) -> str:
    if not e_or_ts:
        return "—"
    ts = e_or_ts if isinstance(e_or_ts, str) else e_or_ts.get("ts")
    return format_ts(ts) if ts else "—"


def _source_ts(source: dict | None) -> str | None:
    """The plain "YYYY-MM-DD HH:MM:SS" timestamp a lineage `source` dict
    (`feature_source`/`gbm_source`/`distillation_source`) actually points to
    — from its own `ts` field where present, else parsed off the leading
    timestamp of the older `label`-only shape (events logged before `ts` was
    split out of the richer picker-dropdown label — every label starts with
    a clean timestamp followed by " — <description>"). `None` if `source`
    itself is missing or carries neither. Shared by `_fmt_built_from`
    (display) and `_source_stale` (the actual staleness check) so a
    pre-migration event with no top-level `ts` is resolved the same way by
    both — a source's `ts` field being optional must not silently make it
    unstale-able forever, just harder to read the timestamp off of."""
    if not source:
        return None
    if source.get("ts"):
        return _fmt_ts(source["ts"])
    label = source.get("label")
    return label.split(" — ")[0] if label else None


def _fmt_built_from(upstream_stage: str, source: dict | None, history: list[str]) -> str:
    if not source:
        return "not recorded (loaded directly from a checkpoint/snapshot, not a fresh run this session)"
    ts = _source_ts(source) or "?"
    position = _position_label(ts, history)
    suffix = f" ({position})" if position else ""
    return f"{upstream_stage}: {ts}{suffix}"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(":material/monitoring: Pricing Agent")
    st.caption("Non-Life Motor — GLM Distillation Dashboard")
    st.divider()

    feat_done = any(e["event"] == "feature_selection_complete" for e in events)
    gbm_done_flag = any(e["event"] == "gbm_complete" for e in events)
    distill_done = any(e["event"] == "glm_distillation_complete" for e in events)
    glm_done = any(e["event"] == "rating_factors" for e in events)

    # Staleness cascades downstream: a stage is stale if its own recorded
    # upstream source is no longer the newest available (_source_stale), OR
    # the upstream stage it was built from is itself stale — one hop of
    # drift anywhere in the ancestry is enough to distrust everything below
    # it, not just the stage that changed. Feature Selection has no
    # upstream, so it's never stale, only pending/done. Own-stale is tracked
    # separately from the cascaded flag so the tooltip can say which one
    # actually happened, rather than a generic "something upstream changed".
    gbm_stale = _source_stale(gbm_ev.get("feature_source") if gbm_ev else None, feature_history)
    distill_own_stale = _source_stale(distill_ev.get("gbm_source") if distill_ev else None, gbm_history)
    distill_stale = distill_own_stale or gbm_stale
    glm_own_stale = _source_stale(rating_ev.get("distillation_source") if rating_ev else None, distill_history)
    glm_stale = glm_own_stale or distill_stale

    st.markdown("**Pipeline Stages**")
    render_pipeline_graph([
        ("Feature Selection", feat_done, "" if feat_done else "Not yet run"),
        (
            "GBM Training", gbm_done_flag and not gbm_stale,
            "Not yet run" if not gbm_done_flag
            else "Stale — Feature Selection has a newer finalized version than this was trained on" if gbm_stale
            else "",
        ),
        (
            "GLM Distillation", distill_done and not distill_stale,
            "Not yet run" if not distill_done
            else "Stale — GBM Training has a newer run than this was finalized against" if distill_own_stale
            else "Stale — GBM Training is itself stale (built from an outdated Feature Selection version)"
            if distill_stale else "",
        ),
        (
            "GLM Fitting", glm_done and not glm_stale,
            "Not yet run" if not glm_done
            else "Stale — GLM Distillation has a newer finalized version than this was fit on" if glm_own_stale
            else "Stale — GLM Distillation is itself stale (built from an outdated upstream version)"
            if glm_stale else "",
        ),
    ])

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
    if st.button(
        "Refresh",
        icon=":material/refresh:",
        help=(
            "Loads the latest saved data right away, instead of waiting a "
            "few seconds for it to appear on its own. Doesn't change or "
            "delete anything — use it after finalizing a step in another "
            "browser tab, or after someone else updates the pipeline, so "
            "you're looking at the current version."
        ),
    ):
        st.cache_data.clear()
        st.rerun()


# ── MAIN TABS ─────────────────────────────────────────────────────────────────

(tab_overview, tab_workbench, tab_gbm, tab_distillation, tab_glm, tab_audit) = st.tabs([
    "Overview",
    "Feature & Grouping Workbench",
    "GBM",
    "GLM Distillation",
    "GLM Results",
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
        c5, c6 = st.columns(2)
        c5.metric("Deviance Explained", f"{rating_ev['deviance_explained']:.1%}")
        c6.metric("Rating Parameters", len(rating_ev.get("rating_factors", [])))

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

    gbm_workbench.render_gbm_control(cfg, CONFIG_DIR / "project_config.yaml")
    st.divider()

    # Prefer the checkpoint (always in sync with the current feature set — cleared
    # on invalidation) over the session-log event, which can be stale or from an
    # older feature set that no longer matches what's checkpointed.
    gbm_output = cfg.get("gbm_output", {})
    checkpoint_interactions = gbm_output.get("interactions")
    feature_importances = gbm_output.get("feature_importances") or (
        gbm_ev.get("feature_importances") if gbm_ev else None
    )

    if not checkpoint_interactions or not feature_importances:
        st.caption("No current GBM results to show — train the GBM above.")
    else:
        st.subheader("Feature Importance (Gain)")

        df_fi = pd.DataFrame(feature_importances).sort_values("importance")
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

        interactions = checkpoint_interactions
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
# GLM DISTILLATION
# ══════════════════════════════════════════════════════════════════════════════

with tab_distillation:
    st.header("GLM Distillation Workbench")
    st.caption(
        "The agent proposes main effects for every approved feature plus pairwise "
        "interactions ranked by the GBM's H-statistics. Review each term below — "
        "include or exclude it, leave a comment — then re-run the agent with your "
        "feedback. Repeat until you finalize; finalizing writes the checkpoint that "
        "the GLM fitting step reads from."
    )
    glm_workbench.render_glm_workbench(cfg, CONFIG_DIR / "glm_config.yaml")


# ══════════════════════════════════════════════════════════════════════════════
# GLM RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_glm:
    st.header("GLM Results — Gamma Log-Link")

    st.subheader("Coefficient Review")
    st.caption(
        "Fits the GLM from the approved distillation terms, then reviews each term's "
        "coefficient sign, significance, and CI. Rejecting a term drops it and refits "
        "automatically — repeat until every remaining term is kept."
    )
    glm_coef_workbench.render_glm_coef_review(cfg, CONFIG_DIR / "glm_config.yaml")
    st.divider()

    if rating_ev:
        c1, c2, c3 = st.columns(3)
        c1.metric("Deviance Explained", f"{rating_ev['deviance_explained']:.2%}")
        c2.metric("Main Effects", len(approved_terms_main))
        c3.metric("Interactions", len(approved_terms_inter))

    formula = glm_cfg.get("glm", {}).get("formula", "") if glm_cfg else ""
    if formula:
        with st.expander("Model formula"):
            st.code(formula, language=None)

    # AIC only means something next to other AICs from models fit on the same
    # target — a single value has no absolute interpretation, unlike Deviance
    # Explained above. So rather than a lone "AIC" tile, every completed fit
    # (each `rating_factors` event — logged once per Fit GLM/refit cycle,
    # only once every remaining term clears coefficient review, never for the
    # intermediate refits inside that loop) gets a row here to compare
    # against. Main Effects/Interactions here are the term set that
    # particular fit actually ran with, which can differ from the tiles
    # above (today's approved glm_config.yaml) if terms changed since.
    rating_events = [e for e in events if e["event"] == "rating_factors"]
    if rating_events:
        with st.expander(f"Fit History ({len(rating_events)})"):
            fit_rows = []
            for e in sorted(rating_events, key=lambda e: e.get("ts", ""), reverse=True):
                n_main, n_inter = _term_counts(e.get("rating_factors", []))
                fit_rows.append({
                    "Timestamp": _fmt_ts(e),
                    "Deviance Explained": f"{e['deviance_explained']:.2%}",
                    "AIC": f"{e['aic']:,.0f}",
                    "Main Effects": n_main,
                    "Interactions": n_inter,
                    "Built from": _fmt_built_from(
                        "GLM Distillation", e.get("distillation_source"), distill_history,
                    ),
                })
            st.dataframe(pd.DataFrame(fit_rows), use_container_width=True, hide_index=True)

    glm_sub = st.tabs(["Main Effects", "Interactions", "Rating Factors"])

    # ── Main Effects ──────────────────────────────────────────────────────────
    with glm_sub[0]:
        if not approved_terms_main:
            st.caption("No main effects in the fitted model.")
        for t in approved_terms_main:
            with st.container(border=True):
                st.markdown(f"**{t['name']}**")
                if t.get("rationale"):
                    st.markdown(f"**Rationale:** {t['rationale']}")
                render_comment_history(_term_note_entries(t))

    # ── Interactions ──────────────────────────────────────────────────────────
    with glm_sub[1]:
        if not approved_terms_inter:
            st.caption("No interactions in the fitted model.")
        for t in sorted(approved_terms_inter, key=lambda t: -(t.get("h_statistic") or 0)):
            with st.container(border=True):
                st.markdown(f"**{t['name']}**")
                st.caption(f"📊 H-statistic: {t.get('h_statistic') or 0:.4f}")
                if t.get("rationale"):
                    st.markdown(f"**Rationale:** {t['rationale']}")
                render_comment_history(_term_note_entries(t))

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
                    "Coefficient": round(c["exp_coef"], 4),
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

            # No color-coding on Coefficient: its magnitude alone isn't
            # informative without the feature's level context (a large value
            # can be an unremarkable base-heavy level or a genuinely large
            # effect) — a fixed color threshold on the raw value would imply
            # a significance it doesn't have.
            st.dataframe(
                df_show,
                column_config={
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                    "Coefficient": st.column_config.NumberColumn(format="%.4f"),
                    "CI Lower": st.column_config.NumberColumn(format="%.4f"),
                    "CI Upper": st.column_config.NumberColumn(format="%.4f"),
                },
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════

with tab_audit:
    st.header("Actuary Decision Audit Trail")

    with st.expander("🔗 Model Lineage", expanded=True):
        if not rating_ev:
            st.caption("No fitted GLM yet — lineage will show once the model is fit.")
        else:
            lineage_rows = [
                {
                    "Stage": "GBM Training", "Timestamp": _fmt_ts(gbm_ev),
                    "Built from": _fmt_built_from(
                        "Feature snapshot", gbm_ev.get("feature_source") if gbm_ev else None, feature_history,
                    ),
                },
                {
                    "Stage": "GLM Distillation (finalized)", "Timestamp": _fmt_ts(distill_ev),
                    "Built from": _fmt_built_from(
                        "GBM run", distill_ev.get("gbm_source") if distill_ev else None, gbm_history,
                    ),
                },
                {
                    "Stage": "GLM Fit", "Timestamp": _fmt_ts(rating_ev),
                    "Built from": _fmt_built_from(
                        "GLM Distillation", rating_ev.get("distillation_source"), distill_history,
                    ),
                },
            ]
            st.dataframe(pd.DataFrame(lineage_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Each row is one completed stage; \"Built from\" names the exact "
                "upstream version it ran against and where that version sits in "
                "its own history — \"latest\" if it was the newest available at "
                "the time, \"2nd latest\" etc. if an older one was deliberately "
                "used instead. Timestamp meanings differ by stage: GBM has no "
                "separate finalize step (every training run is immediately "
                "usable), so its Timestamp is just when training finished; GLM "
                "Distillation's is specifically when it was finalized; GLM Fit's "
                "is when coefficient review completed (every term kept). Feature "
                "snapshots and GLM Distillation versions are real files under "
                "reports/drafts/finalized/, reloadable from each workbench's own "
                "\"Load a saved snapshot\" picker; a GBM run has no such file — "
                "only a session-log entry, reloadable via GLM "
                "Distillation's \"GBM run to distill from\" picker. GLM Fit itself "
                "is never saved as a reloadable version. This is a best-effort "
                "chain (the most recent event of each type, not a strict "
                "cross-reference) — re-running an earlier stage without redoing "
                "the later ones leaves this stale until they catch up."
            )
