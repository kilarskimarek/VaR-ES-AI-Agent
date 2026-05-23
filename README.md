# VaR-ES-AI-Agent

AI agent programmed for calculating **Value at Risk (VaR)** and **Expected Shortfall (ES)** with interpretation.

## Specific Goals
- Estimate VaR and ES of a portfolio
- Compare different VaR/ES methods
- Detect estimation instability
- Interpret results using an LLM (Llama 3 via Ollama)

## Input Data
The agent accepts a `.csv` file with closing prices. The file must contain a `Date` column and a `Close` column.

> Example data used during development: WIG20 index closing prices (02.01.2025 – 23.01.2026).

## Ollama AI
- Download and install Ollama
- Choose model: Llama 3
- Start the Ollama server

## Running App

streamlit run _AI_Agent1_VaR_ES.py

After the website opens, upload CSV. Agent gonna analyse it and give proper feedback.

## Possible Future Development
- Rolling window VaR/ES estimation
- Backtesting module
- Additional estimation methods
- Multi-asset portfolio support
