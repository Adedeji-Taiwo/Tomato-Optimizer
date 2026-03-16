import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pyomo.environ as pyo

from data_generator import tomato_data
from optimiser import build_model, solve_model, extract_results

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TomatoPro | Stochastic Optimiser",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  –  refined dark-green agri theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --tomato:    #e8392a;
    --tomato-lt: #ff6b5b;
    --green:     #1a6b3a;
    --green-lt:  #2d9a56;
    --gold:      #f5a623;
    --bg:        #0e1610;
    --card:      #141f17;
    --border:    #243328;
    --text:      #e8ede9;
    --muted:     #7a9980;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0b130d !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ── */
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--tomato-lt) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    background: var(--card) !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--tomato), #c0281a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(232,57,42,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(232,57,42,0.5) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--tomato-lt) !important;
    border-bottom: 2px solid var(--tomato) !important;
}

/* ── Dividers & misc ── */
hr { border-color: var(--border) !important; }
.stAlert { border-radius: 10px !important; }

/* ── Info boxes ── */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--green-lt);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}
.info-card h4 { color: var(--gold); margin: 0 0 0.4rem; font-size: 0.85rem;
                text-transform: uppercase; letter-spacing: 0.08em; }
.info-card p  { margin: 0; color: var(--text); font-size: 0.9rem; line-height: 1.6; }

.section-label {
    color: var(--muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #ff6b5b, #f5a623);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.3rem;
}

.pill {
    display: inline-block;
    background: rgba(232,57,42,0.15);
    color: var(--tomato-lt);
    border: 1px solid rgba(232,57,42,0.3);
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 0.3rem;
}

.scenario-badge-good    { color:#2d9a56; font-weight:700; }
.scenario-badge-average { color:#f5a623; font-weight:700; }
.scenario-badge-bad     { color:#e8392a; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly dark theme helper
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,31,23,0.6)",
    font=dict(family="DM Sans", color="#e8ede9"),
    xaxis=dict(gridcolor="#243328", linecolor="#243328"),
    yaxis=dict(gridcolor="#243328", linecolor="#243328"),
)
COLORS = {
    "Fresh": "#2d9a56",
    "Paste": "#f5a623",
    "Dried": "#e8392a",
}
SCEN_COLORS = {"Good": "#2d9a56", "Average": "#f5a623", "Bad": "#e8392a"}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:2.5rem;'>🍅</div>
      <div style='font-family:"Playfair Display",serif; font-size:1.3rem;
                  font-weight:900; color:#ff6b5b;'>TomatoPro</div>
      <div style='color:#7a9980; font-size:0.72rem; letter-spacing:0.1em;
                  text-transform:uppercase; margin-top:0.2rem;'>
        Stochastic Portfolio Optimiser
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Optimisation Parameters</div>',
                unsafe_allow_html=True)

    risk_weight = st.slider(
        "Risk Aversion Weight",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="0 = maximise expected profit only.  1 = fully minimise downside risk (CVaR)."
    )
    alpha = st.slider(
        "CVaR Tail Probability (α)",
        min_value=0.05, max_value=0.30, value=0.10, step=0.01,
        help="Lower α = more conservative. α=0.10 protects against the worst 10% of outcomes."
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Solver</div>',
                unsafe_allow_html=True)
    solver_choice = st.selectbox(
        "LP Solver", ["glpk", "cbc", "highs"], index=0)

    st.markdown("---")
    run_btn = st.button("▶  Run Optimisation", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#7a9980; font-size:0.72rem; line-height:1.7;'>
    <b style='color:#e8ede9;'>Model type:</b> Two-stage stochastic LP<br>
    <b style='color:#e8ede9;'>Risk measure:</b> CVaR (Rockafellar & Uryasev)<br>
    <b style='color:#e8ede9;'>Solver:</b> GLPK / CBC / HiGHS<br>
    <b style='color:#e8ede9;'>Framework:</b> Pyomo + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main area – tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_data, tab_results, tab_risk, tab_about = st.tabs([
    "Overview",
    "Input Data",
    "Optimisation Results",
    "Risk Analysis",
    "Methodology",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("""
    <div class="hero-title">TomatoPro</div>
    <div class="hero-sub">
        A stochastic decision-support tool for tomato processing portfolio optimisation &nbsp;·&nbsp; Nigeria
    </div>
    <div style='margin-top:0.6rem; color:#7a9980; font-size:0.82rem;'>
        Advanced Analytics for Agribusiness &nbsp;·&nbsp; MSc Agribusiness &amp; Innovation &nbsp;·&nbsp; UM6P
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
          <h4>The Problem</h4>
          <p>Tomato processors in Nigeria operate under compounding seasonal uncertainty:
          yields fluctuate between 10 and 20 t/ha depending on rainfall, spoilage rates
          reach 20% in poor seasons, and prices across three product channels move
          independently. Deterministic planning systematically underperforms.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
          <h4>The Approach</h4>
          <p>A <b>two-stage stochastic linear programme</b> models three rainfall scenarios
          (Good / Average / Bad). Stage 1 reserves processing capacity before the season.
          Stage 2 adapts production once the scenario is realised. A CVaR objective
          (Rockafellar &amp; Uryasev, 2000) allows explicit control over downside risk.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
          <h4>Three Products</h4>
          <p>
          <b>Fresh Market</b> — 1:1 conversion ratio, ₦120k/t base price, highest volatility.<br><br>
          <b>Tomato Paste</b> — 5 t fresh per tonne output, ₦250k/t, more stable demand.<br><br>
          <b>Dried Tomatoes</b> — 8 t fresh per tonne output, ₦400k/t, export-oriented.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Scenario overview cards
    st.markdown("#### Scenario Structure")
    c1, c2, c3 = st.columns(3)
    scenarios_meta = [
        ("Good Season", "🌧️", "30%", "20 t/ha", "5%", "#2d9a56"),
        ("Average Season", "⛅", "50%", "15 t/ha", "10%", "#f5a623"),
        ("Bad Season", "☀️", "20%", "10 t/ha", "20%", "#e8392a"),
    ]
    for col, (name, icon, prob, yld, spill, clr) in zip([c1, c2, c3], scenarios_meta):
        with col:
            st.markdown(f"""
            <div style="background:var(--card); border:1px solid {clr}44;
                        border-top:4px solid {clr}; border-radius:12px;
                        padding:1.2rem; text-align:center;">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                          color:{clr}; font-weight:700; margin:0.4rem 0;">{name}</div>
              <div style="color:var(--muted); font-size:0.78rem; line-height:2;">
                Probability: <b style="color:var(--text)">{prob}</b><br>
                Yield: <b style="color:var(--text)">{yld}</b><br>
                Spoilage: <b style="color:var(--text)">{spill}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Adjust **Risk Aversion** (λ) and **CVaR tail probability** (α) in the sidebar, then click **▶ Run Optimisation** to generate the recommended allocation plan.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – INPUT DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📊 Input Data Explorer")
    st.markdown(
        "All data is synthetically generated but calibrated to Nigerian agriculture statistics.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Available Tomatoes by Scenario (tons)")
        avail_df = pd.DataFrame.from_dict(
            tomato_data['available'], orient='index', columns=['Tonnes Available']
        ).round(1)
        avail_df.index.name = "Scenario"
        st.dataframe(avail_df, use_container_width=True)

        st.markdown("#### Market Prices (₦ / ton)")
        price_rows = []
        for s in tomato_data['scenarios']:
            row = {'Scenario': s}
            row.update({p: f"₦{tomato_data['prices'][s][p]:,.0f}"
                        for p in ['Fresh', 'Paste', 'Dried']})
            price_rows.append(row)
        st.dataframe(pd.DataFrame(price_rows).set_index(
            'Scenario'), use_container_width=True)

    with col_b:
        st.markdown("#### Processing Capacity & Conversion")
        cap_data = {
            'Product': ['Fresh', 'Paste', 'Dried'],
            'Max Capacity (t/wk)': [100, 50, 20],
            'Fresh Input per Output Tonne': [1, 5, 8],
            'Processing Cost (₦/t)': ['₦5,000', '₦50,000', '₦80,000'],
        }
        st.dataframe(pd.DataFrame(cap_data).set_index(
            'Product'), use_container_width=True)

        st.markdown("#### Scenario Probabilities & Quality")
        qual_data = {
            'Scenario': ['Good', 'Average', 'Bad'],
            'Probability': ['30%', '50%', '20%'],
            'Brix (%)': [6.0, 5.0, 4.0],
            'Spoilage (%)': [5, 10, 20],
        }
        st.dataframe(pd.DataFrame(qual_data).set_index(
            'Scenario'), use_container_width=True)

    # Price distribution chart
    st.markdown("#### Price Distribution Across Scenarios")
    fig_prices = go.Figure()
    for prod, clr in COLORS.items():
        vals = [tomato_data['prices'][s][prod] /
                1000 for s in tomato_data['scenarios']]
        fig_prices.add_trace(go.Bar(
            name=prod, x=tomato_data['scenarios'], y=vals,
            marker_color=clr,
            text=[f"₦{v:.0f}k" for v in vals],
            textposition='outside',
        ))
    fig_prices.update_layout(
        barmode='group',
        title="Market Prices by Scenario (₦ thousands per tonne)",
        **PLOTLY_THEME,
        legend=dict(orientation='h', y=1.12),
        height=380,
    )
    st.plotly_chart(fig_prices, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: run model (cached by parameters)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_optimisation(risk_weight, alpha, solver):
    model = build_model(tomato_data, risk_weight=risk_weight, alpha=alpha)
    model, _ = solve_model(model, solver=solver)
    results = extract_results(model, tomato_data)
    # Extract VaR if present
    if hasattr(model, 'VaR'):
        results['VaR_val'] = pyo.value(model.VaR)
        results['alpha'] = alpha
        expected_slack = sum(
            tomato_data['probabilities'][i] * pyo.value(model.CVaR_slack[s])
            for i, s in enumerate(tomato_data['scenarios'])
        )
        results['CVaR_val'] = results['VaR_val'] - \
            (1.0 / alpha) * expected_slack
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if run_btn:
        with st.spinner("Solving two-stage stochastic LP…"):
            try:
                res = run_optimisation(risk_weight, alpha, solver_choice)
                st.session_state['results'] = res
                st.session_state['rw'] = risk_weight
                st.session_state['al'] = alpha
            except Exception as e:
                st.error(
                    f"**Solver error:** {e}\n\nEnsure GLPK is installed: `sudo apt-get install glpk-utils` (Linux) or `brew install glpk` (Mac).")
                st.stop()

    if 'results' not in st.session_state:
        st.info("Run the optimisation from the sidebar to see results here.")
    else:
        res = st.session_state['results']
        rw = st.session_state['rw']
        al = st.session_state['al']

        st.markdown("### Optimal Decision Plan")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected Profit", f"₦{res['expected_profit']/1e6:.2f}M")
        total_reserve = sum(res['reserves'].values())
        k2.metric("Total Reserved Capacity", f"{total_reserve:.1f} t")
        best_scenario = max(res['scenario_profits'],
                            key=res['scenario_profits'].get)
        k3.metric("Best Scenario", best_scenario,
                  f"₦{res['scenario_profits'][best_scenario]/1e6:.2f}M")
        worst_scenario = min(res['scenario_profits'],
                             key=res['scenario_profits'].get)
        k4.metric("Worst Scenario", worst_scenario,
                  f"₦{res['scenario_profits'][worst_scenario]/1e6:.2f}M")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1.6])

        with col_left:
            st.markdown("#### Stage 1 – Capacity Reservation")
            st.markdown("*(Decided before the season is known)*")
            res_df = pd.DataFrame.from_dict(
                res['reserves'], orient='index', columns=['Reserved (t)']
            ).round(2)
            res_df.index.name = "Product"
            st.dataframe(res_df, use_container_width=True)

            st.markdown("#### Stage 2 – Production by Scenario")
            st.markdown("*(Adapted once rainfall / yield is observed)*")
            prod_display = res['production'].copy().astype(float).round(2)
            st.dataframe(prod_display.style.background_gradient(
                cmap='RdYlGn', axis=None), use_container_width=True)

        with col_right:
            # Production bar chart
            fig_prod = go.Figure()
            for prod, clr in COLORS.items():
                fig_prod.add_trace(go.Bar(
                    name=prod,
                    x=tomato_data['scenarios'],
                    y=res['production'][prod].values,
                    marker_color=clr,
                    text=[f"{v:.1f}t" for v in res['production'][prod].values],
                    textposition='outside',
                ))
            fig_prod.update_layout(
                barmode='group',
                title="Optimal Production Allocation by Scenario (tonnes)",
                **PLOTLY_THEME,
                legend=dict(orientation='h', y=1.12),
                height=400,
            )
            st.plotly_chart(fig_prod, use_container_width=True)

        # Scenario profit comparison
        st.markdown("#### Profit Across Scenarios")
        sc_profs = res['scenario_profits']
        fig_profit = go.Figure()
        fig_profit.add_trace(go.Bar(
            x=list(sc_profs.keys()),
            y=[v / 1e6 for v in sc_profs.values()],
            marker_color=[SCEN_COLORS[s] for s in sc_profs.keys()],
            text=[f"₦{v/1e6:.2f}M" for v in sc_profs.values()],
            textposition='outside',
        ))
        fig_profit.add_hline(
            y=res['expected_profit'] / 1e6,
            line_dash='dash', line_color='#f5a623',
            annotation_text=f"E[Profit] = ₦{res['expected_profit']/1e6:.2f}M",
            annotation_position="top right",
        )
        fig_profit.update_layout(
            title="Scenario Profits vs. Expected Profit (₦ millions)",
            **PLOTLY_THEME,
            showlegend=False,
            height=360,
        )
        st.plotly_chart(fig_profit, use_container_width=True)

        # Revenue waterfall
        st.markdown("#### Revenue Breakdown – Expected Values")
        prods = list(COLORS.keys())
        rev_vals = []
        cost_vals = []
        for p in prods:
            rev = sum(
                tomato_data['probabilities'][i] *
                tomato_data['prices'][s][p] * float(res['sales'][p][s])
                for i, s in enumerate(tomato_data['scenarios'])
            )
            cost = sum(
                tomato_data['probabilities'][i] *
                tomato_data['proc_cost'][p] * float(res['production'][p][s])
                for i, s in enumerate(tomato_data['scenarios'])
            )
            rev_vals.append(rev / 1e6)
            cost_vals.append(cost / 1e6)

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(name='Revenue', x=prods,
                                y=rev_vals, marker_color=[COLORS[p] for p in prods]))
        fig_wf.add_trace(go.Bar(name='Processing Cost', x=prods,
                                y=[-c for c in cost_vals],
                                marker_color='rgba(150,150,150,0.5)'))
        fig_wf.update_layout(
            barmode='relative',
            title="Expected Revenue vs Processing Cost by Product (₦M)",
            **PLOTLY_THEME,
            legend=dict(orientation='h', y=1.12),
            height=360,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Download
        csv = res['production'].to_csv()
        st.download_button(
            "⬇️  Download Production Plan (CSV)",
            data=csv,
            file_name="tomatopro_production_plan.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("### Risk–Return Analysis")

    if 'results' not in st.session_state:
        st.info("Run the optimisation first to see risk metrics.")
    else:
        res = st.session_state['results']
        rw = st.session_state['rw']
        al = st.session_state['al']

        # Risk metric cards
        if 'CVaR_val' in res:
            r1, r2, r3 = st.columns(3)
            r1.metric("Expected Profit", f"₦{res['expected_profit']/1e6:.2f}M")
            r2.metric("Value at Risk (VaR)", f"₦{res['VaR_val']/1e6:.2f}M",
                      help="Minimum profit above the α-worst scenarios.")
            r3.metric("CVaR (Expected Shortfall)", f"₦{res['CVaR_val']/1e6:.2f}M",
                      help=f"Average profit in the worst {int(al*100)}% of scenarios.")
        else:
            st.metric("Expected Profit (Risk-Neutral)",
                      f"₦{res['expected_profit']/1e6:.2f}M")
            st.info(
                "CVaR metrics are computed when Risk Aversion > 0.  Set the slider and re-run.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Efficient frontier simulation
        st.markdown("#### Efficient Frontier: Expected Profit vs. CVaR")
        st.markdown("Each point represents a different risk-aversion parameter (sweep from 0 → 1). "
                    "This reveals the full trade-off curve.")

        with st.spinner("Computing frontier (scanning risk weights)…"):
            frontier_pts = []
            for rw_sweep in np.linspace(0.0, 1.0, 15):
                try:
                    r_sw = run_optimisation(rw_sweep, al, solver_choice)
                    ep = r_sw['expected_profit']
                    if 'CVaR_val' in r_sw:
                        cv = r_sw['CVaR_val']
                    else:
                        sc_p = r_sw['scenario_profits']
                        probs = tomato_data['probabilities']
                        sorted_profits = sorted(sc_p.values())
                        # simple CVaR approximation
                        cv = sorted_profits[0]
                    frontier_pts.append({'risk_weight': rw_sweep,
                                         'expected_profit': ep / 1e6,
                                         'cvar': cv / 1e6})
                except Exception:
                    pass

        if frontier_pts:
            df_frontier = pd.DataFrame(frontier_pts)
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=df_frontier['cvar'],
                y=df_frontier['expected_profit'],
                mode='lines+markers',
                line=dict(color='#ff6b5b', width=2.5),
                marker=dict(size=8, color=df_frontier['risk_weight'],
                            colorscale='RdYlGn_r', showscale=True,
                            colorbar=dict(title='Risk Weight', tickfont=dict(color='#e8ede9'))),
                hovertemplate='CVaR: ₦%{x:.2f}M<br>E[Profit]: ₦%{y:.2f}M<extra></extra>',
            ))
            # Mark current solution
            cur_ep = res['expected_profit'] / 1e6
            cur_cv = res.get('CVaR_val', res['expected_profit']) / 1e6
            fig_ef.add_trace(go.Scatter(
                x=[cur_cv], y=[cur_ep],
                mode='markers', marker=dict(size=14, color='#f5a623',
                                            symbol='star', line=dict(color='white', width=1.5)),
                name='Current Solution',
                hovertemplate='★ Your Selection<br>CVaR: ₦%{x:.2f}M<br>E[Profit]: ₦%{y:.2f}M<extra></extra>',
            ))
            fig_ef.update_layout(
                title="Efficient Frontier (Risk–Return Trade-off)",
                xaxis_title="CVaR – Expected Profit in Worst Scenarios (₦M)",
                yaxis_title="Expected Profit (₦M)",
                **PLOTLY_THEME,
                height=450,
                showlegend=True,
                legend=dict(orientation='h', y=1.08),
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        # Scenario profit distribution
        st.markdown("#### Profit Distribution Across Scenarios")
        sc_p = res['scenario_profits']
        probs_list = tomato_data['probabilities']
        fig_dist = go.Figure()
        for (s, profit), prob, clr in zip(sc_p.items(), probs_list, SCEN_COLORS.values()):
            fig_dist.add_trace(go.Bar(
                name=s, x=[s],
                y=[profit / 1e6],
                marker_color=clr,
                text=f"₦{profit/1e6:.2f}M<br>(p={prob})",
                textposition='outside',
                width=0.4,
            ))
        if 'CVaR_val' in res:
            fig_dist.add_hline(y=res['CVaR_val'] / 1e6,
                               line_dash='dash', line_color='#e8392a',
                               annotation_text=f"CVaR = ₦{res['CVaR_val']/1e6:.2f}M",
                               annotation_position="top right")
        fig_dist.add_hline(y=res['expected_profit'] / 1e6,
                           line_dash='dot', line_color='#f5a623',
                           annotation_text=f"E[Profit] = ₦{res['expected_profit']/1e6:.2f}M",
                           annotation_position="bottom right")
        fig_dist.update_layout(
            title="Profit by Scenario with Risk Thresholds",
            yaxis_title="Profit (₦M)",
            **PLOTLY_THEME,
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### Methodology")
    st.markdown(
        "This project was developed as a group course capstone for **Advanced Analytics for "
        "Agribusiness** (MSc Agribusiness & Innovation, UM6P). The objective was to apply "
        "stochastic programming to a realistic agri-food processing decision problem, and to "
        "deliver the resulting model as an interactive decision-support tool accessible to "
        "non-technical stakeholders."
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(r"""
#### Problem Framing

Tomato processing in Nigeria presents a canonical example of a **perishable commodity
allocation problem under uncertainty**. A processor contracts a fixed area of farmland
(here, 100 ha) and must commit processing capacity — equipment scheduling, labour, cold
storage — before the harvest outcome is known. The realised scenario then determines how
much raw material is actually available and at what quality.

Three sources of uncertainty interact:

1. **Supply yield** — seasonal rainfall drives yields between 10 t/ha (drought) and
   20 t/ha (good rainfall), a twofold variation on the same contracted area.
2. **Spoilage** — post-harvest losses range from 5% to 20% depending on ambient
   temperature and road quality, both correlated with season.
3. **Market prices** — fresh-market, paste, and dried-tomato prices are driven by
   different supply chains. Fresh prices are highly volatile (local market saturation);
   dried prices are more stable (export demand).

Static planning — fixing production targets based on average expectations — is
suboptimal under all three scenarios. Stochastic programming provides a principled
framework for making robust decisions before uncertainty resolves.

#### Two-Stage Stochastic Linear Programme

The model follows the standard two-stage recourse formulation
(Birge & Louveaux, 2011):

**Stage 1 — here-and-now decisions** (made before the season):

> Reserve processing capacity $x_p \geq 0$ for each product $p$.

**Stage 2 — wait-and-see decisions** (made after observing scenario $s$):

> Choose production $q_{s,p}$ and sales $v_{s,p}$ within reserved capacity and
> available tomato supply.

The deterministic equivalent programme maximises:

> $\sum_{s} \pi_s \left[ \sum_p \left( r_{s,p} \cdot v_{s,p} - c_p \cdot q_{s,p} \right) \right]$

subject to material balance, capacity, and non-negativity constraints (see table).
        """)

    with c2:
        st.markdown(r"""
#### Model Constraints

| Constraint | Expression |
|---|---|
| Material balance | $\sum_p q_{s,p} \cdot \delta_p \leq A_s \quad \forall s$ |
| Capacity reservation | $q_{s,p} \leq x_p \quad \forall s, p$ |
| Plant capacity | $q_{s,p} \leq \kappa_p \quad \forall s, p$ |
| Sales bound | $v_{s,p} \leq q_{s,p} \quad \forall s, p$ |
| Non-negativity | $x_p, q_{s,p}, v_{s,p} \geq 0$ |

where $\delta_p$ is the fresh-tomato input required per tonne of product $p$,
$A_s$ is available supply in scenario $s$, and $\kappa_p$ is plant capacity.

#### Risk-Averse Objective

When the decision-maker wishes to protect against downside outcomes, the objective
is augmented with **Conditional Value-at-Risk** (CVaR) at confidence level $\alpha$:

> $\max \; (1-\lambda) \cdot \mathbb{E}[\Pi] \; + \; \lambda \cdot \text{CVaR}_\alpha[\Pi]$

CVaR$_\alpha$ is the expected profit in the worst $\alpha$-fraction of scenarios.
Following Rockafellar & Uryasev (2000), it is linearised exactly as:

> $\text{CVaR}_\alpha = \eta - \dfrac{1}{\alpha} \sum_s \pi_s \cdot \zeta_s$

where $\eta$ (VaR) and $\zeta_s \geq 0$ (slack variables) are added to the LP.
This preserves linearity — no integer variables or scenario-tree enumeration required.

$\lambda = 0$ recovers the risk-neutral expected-profit maximiser;
$\lambda = 1$ maximises CVaR exclusively.

#### Implementation Notes

The model is implemented in **Pyomo** (Hart et al., 2017) and solved with **GLPK**.
Scenario data are synthetically generated but calibrated to published Nigerian
agricultural statistics (FAOSTAT, 2023). The interface is built in **Streamlit**
with **Plotly** for interactive visualisation.

The efficient frontier (Risk Analysis tab) is produced by parametric sweep
over $\lambda \in [0, 1]$, solving a separate LP at each point.
        """)

    st.markdown("---")
    st.markdown("""
#### References

Birge, J. R., & Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer.

Hart, W. E., Laird, C. D., Watson, J.-P., Woodruff, D. L., Hackebeil, G. A., Nicholson, B. L., & Siirola, J. D. (2017). *Pyomo — Optimization Modeling in Python* (2nd ed.). Springer.

FAOSTAT (2023). *Crops and livestock products: Nigeria*. Food and Agriculture Organization of the United Nations. https://www.fao.org/faostat

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.
    """)

    st.markdown("---")
    st.markdown(
        "<div style='color:#7a9980; font-size:0.78rem;'>"
        "Advanced Analytics for Agribusiness &nbsp;·&nbsp; "
        "MSc Agribusiness &amp; Innovation &nbsp;·&nbsp; UM6P &nbsp;·&nbsp; "
        "Built with Python, Pyomo, Streamlit, and Plotly."
        "</div>",
        unsafe_allow_html=True,
    )
