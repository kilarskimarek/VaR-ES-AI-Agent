import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
from arch import arch_model
import requests  

# Założenia
ALPHA = 0.99

RISK_THRESHOLDS = {
    "es_high": -0.07,
    "model_spread": 0.02
}

# Załadowanie danych
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
    return df

def calculate_returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().dropna()

# Ryzyko modelu
def historical_var_es(pnl: pd.Series, alpha=ALPHA):
    var = np.quantile(pnl, 1 - alpha)
    es = pnl[pnl <= var].mean()
    return var, es

def garch_var_es(pnl: pd.Series, alpha=ALPHA):
    model = arch_model(pnl, vol="Garch", p=1, q=1)
    res = model.fit(disp="off")
    sigma = res.conditional_volatility.iloc[-1]
    mu = pnl.mean()
    var = norm.ppf(1 - alpha, mu, sigma)
    es = mu - sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha)
    return var, es

def monte_carlo_var_es(mu: float, sigma: float, alpha=ALPHA, n=100_000):
    sims = np.random.normal(mu, sigma, n)
    var = np.quantile(sims, 1 - alpha)
    es = sims[sims <= var].mean()
    return var, es

def collect_var_es(pnl: pd.Series):
    return {
        "historical": historical_var_es(pnl),
        "monte_carlo": monte_carlo_var_es(pnl.mean(), pnl.std()),
        "garch": garch_var_es(pnl)
    }

# Zagregowane rezultaty
def aggregate_results(results: dict):
    vars_ = [v for v, _ in results.values()]
    es_ = [es for _, es in results.values()]
    return {
        "var_min": min(vars_),
        "var_max": max(vars_),
        "es_avg": np.mean(es_),
        "spread": np.std(vars_)
    }

def build_portfolio_profile(metrics: dict):
    return {
        "tail_risk": (
            "high" if metrics["es_avg"] < RISK_THRESHOLDS["es_high"] else "normal"
        ),
        "model_risk": (
            "high" if metrics["spread"] > RISK_THRESHOLDS["model_spread"] else "low"
        )
    }

# Programowanie LLM
def build_prompt(profile: dict, metrics: dict):
    return f"""
You are a financial risk analyst.

Metrics:
- Average ES (99%): {metrics['es_avg']:.4f}
- Minimum VaR: {metrics['var_min']:.4f}
- Maximum VaR: {metrics['var_max']:.4f}
- VaR spread (model risk): {metrics['spread']:.4f}

Portfolio profile:
- Tail risk: {profile['tail_risk']}
- Model risk: {profile['model_risk']}

Please provide a brief interpretation of these results for the risk manager.
"""

# Programowanie interpretacji szykowanej przez agenta przy pomocy Ollama AI
def risk_interpretation(prompt: str) -> str:
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60)
        
        if response.status_code == 200:
            return response.json()['response']
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama."
    except Exception as e:
        return f"Error getting interpretation: {str(e)}"

# Budowa frontendu
st.set_page_config(page_title="VaR/ES Risk Agent", layout="centered")
st.title("Investment Risk Agent")
st.write("Upload a CSV file")

# Sprawdzenie czy Ollama działa
try:
    test_response = requests.get('http://localhost:11434/api/tags', timeout=2)
    if test_response.status_code == 200:
        st.success("Ollama is running and ready!")
    else:
        st.warning("Ollama connection issue.")
except:
    st.error("Ollama is not running.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        # Dane
        df = load_data(uploaded_file)
        pnl = calculate_returns(df)

        # Modele
        results = collect_var_es(pnl)
        metrics = aggregate_results(results)
        profile = build_portfolio_profile(metrics)

        # Metryki
        st.subheader("Risk Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Average ES (99%)", f"{metrics['es_avg']:.2%}")
        c2.metric("VaR Spread", f"{metrics['spread']:.2%}")
        c3.metric("Tail Risk", profile["tail_risk"])

        st.subheader("Portfolio Profile")
        st.write(f"- Tail risk: {profile['tail_risk']}")
        st.write(f"- Model risk: {profile['model_risk']}")

        st.subheader("Agent Interpretation")
        st.info("Powered by Llama 3")
        
        prompt = build_prompt(profile, metrics)

        with st.spinner("The agent is analyzing risk..."):
            interpretation = risk_interpretation(prompt)

        st.text_area("Analytical Comment", interpretation, height=300)

        st.subheader("Model Details")
        for model, (var, es) in results.items():
            st.write(f"{model} → VaR: {var:.4f}, ES: {es:.4f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.text(traceback.format_exc())