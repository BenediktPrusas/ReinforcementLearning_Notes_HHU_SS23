import pandas as pd
import streamlit as st

#st.markdown(
#    """
#    | Algorithms              | Model-based vs Model-free | On-policy vs Off-policy | Deep vs Shallow | Sample-based vs Tree-based | Tabular vs Function Approximation | Target Function Q vs V | Update Formula | Paradigm | Control vs Prediction | TD-Based |
#|-------------------------|---------------------------|-------------------------|-----------------|----------------------------|-----------------------------------|------------------------|----------------|----------|----------------------|----------|
#| Monte Carlo Control     | Model-free                | On-policy               | Deep     | Sample-based                | Can be both                        | Q                      | Q(s, a) &larr; Q(s, a) + α[Gt - Q(s, a)] | Monte Carlo | Control | No       |
#| Monte Carlo Policy Evaluation | Model-free | On-policy | Deep | Sample-based | Can be both | V | V(s) &larr; V(s) + α[Gt - V(s)] | Monte Carlo | Prediction | No       |
#| SARSA                 | Model-free                | On-policy               | Shallow     | Sample-based                | Can be both                        | Q                      | Q(s, a) &larr; Q(s, a) + α[R + γQ(s', a') - Q(s, a)] | Temporal Difference | Can be both | Yes      |
#| Q-Learning             | Model-free                | Off-policy              | Shallow     | Sample-based                | Can be both                        | Q                      | Q(s, a) &larr; Q(s, a) + α[R + γ max_a' Q(s', a') - Q(s, a)] | Temporal Difference | Can be both | Yes      |
#| Q-Value Iteration      | Model-based               | Off-policy              | Shallow     | Tree-based                  | Can be both                        | Q                      | Q(s, a) &larr; Σ_s' P(s'|s, a) [R + γ max_a' Q(s', a')] | Dynamic Programming | Control | No       |
#| Q-Policy Iteration     | Model-based               | Can be both             | Shallow     | Tree-based                  | Can be both                        | Q                      | Q(s, a) &larr; Σ_s' P(s'|s, a) [R + γ Q(s', π(s'))] | Dynamic Programming | Control | No       |
#| TD(0)                  | Model-free                | On-policy               | Shallow     | Sample-based                | Can be both                        | Can be both            | V(s) &larr; V(s) + α[R + γV(s') - V(s)] (for V) <br> Q(s, a) &larr; Q(s, a) + α[R + γQ(s', a') - Q(s, a)] (for Q) | Temporal Difference | Can be both | Yes      |
#| Iterative Policy Evaluation | Model-based | On-policy | Deep | Tree-based | Can be both | V | V(s) &larr; Σ_a π(a&#124;s) Σ_s',r P(s', r&#124;s, a) [r + γV(s')] | Dynamic Programming | Prediction | No       |
#
#    """
#)
st.warning("This is just ChatGpt autoput, i didn't verify the correctness of the table.")

data = {
    "Algorithms": ["Monte Carlo Control", "Monte Carlo Policy Evaluation", "SARSA", "Q-Learning", "Q-Value Iteration", "Q-Policy Iteration", "TD(0)", "Iterative Policy Evaluation"],
    "Model-based vs Model-free": ["Model-free", "Model-free", "Model-free", "Model-free", "Model-based", "Model-based", "Model-free", "Model-based"],
    "On-policy vs Off-policy": ["On-policy", "On-policy", "On-policy", "Off-policy", "Off-policy", "Can be both", "On-policy", "On-policy"],
    "Deep vs Shallow": ["Deep", "Deep", "Shallow", "Shallow", "Shallow", "Shallow", "Shallow", "Deep"],
    "Sample-based vs Tree-based": ["Sample-based", "Sample-based", "Sample-based", "Sample-based", "Tree-based", "Tree-based", "Sample-based", "Tree-based"],
    "Target Function Q vs V": ["Q", "V", "Q", "Q", "Q", "Q", "Can be both", "V"],
    "Update Formula": [
        "Q(s, a) ← Q(s, a) + α[Gt - Q(s, a)]",
        "V(s) ← V(s) + α[Gt - V(s)]",
        "Q(s, a) ← Q(s, a) + α[R + γQ(s', a') - Q(s, a)]",
        "Q(s, a) ← Q(s, a) + α[R + γ max_a' Q(s', a') - Q(s, a)]",
        "Q(s, a) ← Σ_s' P(s'|s, a) [R + γ max_a' Q(s', a')]",
        "Q(s, a) ← Σ_s' P(s'|s, a) [R + γ Q(s', π(s'))]",
        "V(s) ← V(s) + α[R + γV(s') - V(s)] (for V) \n Q(s, a) ← Q(s, a) + α[R + γQ(s', a') - Q(s, a)] (for Q)",
        "V(s) ← Σ_a π(a|s) Σ_s',r P(s', r|s, a) [r + γV(s')]"
    ],
    "Paradigm": ["Monte Carlo", "Monte Carlo", "Temporal Difference", "Temporal Difference", "Dynamic Programming", "Dynamic Programming", "Temporal Difference", "Dynamic Programming"],
    "Control vs Prediction": ["Control", "Prediction", "Can be both", "Can be both", "Control", "Control", "Can be both", "Prediction"],
}

df = pd.DataFrame(data)
df = df.set_index("Algorithms")

st.dataframe(df, use_container_width=True)