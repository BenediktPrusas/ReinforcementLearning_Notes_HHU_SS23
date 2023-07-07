import pandas as pd
import streamlit as st

try:
    st.set_page_config(layout="wide", page_title="Reinforcement Learning Lecture Notes")
except:
    pass

# Monte Carlo ---------------------------------------------------------------
algo_mcc = {
    "Algorithm": "Monte Carlo Control",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "Deep",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Q",
    "Update Formula": "Q(s, a) ← Q(s, a) + α[Gt - Q(s, a)]",
    "Paradigm": "Monte Carlo",
}

algo_mcpe = {
    "Algorithm": "Monte Carlo Policy Evaluation",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "Deep",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← V(s) + α[Gt - V(s)]",
    "Paradigm": "Monte Carlo",
}

algo_mcc_importance_sampling = {
    "Algorithm": "Monte Carlo Control with importance sampling",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "Off-policy",
    "Deep vs Shallow Update": "Deep",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Q",
    "Update Formula": "Q(s, a) ← Q(s, a) + α[ρt*Gt - Q(s, a)]",
    "Paradigm": "Monte Carlo",
}

algo_mcpe_importance_sampling = {
    "Algorithm": "Monte Carlo Policy Evaluation with importance sampling",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "Off-policy",
    "Deep vs Shallow Update": "Deep",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← V(s) + α[ρt*Gt - V(s)]",
    "Paradigm": "Monte Carlo",
}
# Dynamic Programming ---------------------------------------------------------------

algo_iterative_policy_evaluation = {
    "Algorithm": "Iterative Policy Evaluation",
    "Model-based vs Model-free": "Model-based",
    "On-policy vs Off-policy": "Learns from the model not experience",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Tree-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← Σ_a π(a|s) Σ_s',r P(s', r|s, a) [r + γV(s')]",
    "Paradigm": "Dynamic Programming",
}

algo_value_iteration = {
    "Algorithm": "Value Iteration",
    "Model-based vs Model-free": "Model-based",
    "On-policy vs Off-policy": "Learns from the model not experience",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Tree-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← max_a Σ_s',r P(s', r|s, a) [r + γV(s')]",
    "Paradigm": "Dynamic Programming",
}

algo_policy_iteration = {
    "Algorithm": "Policy Iteration",
    "Model-based vs Model-free": "Model-based",
    "On-policy vs Off-policy": "Learns from the model not experience",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Tree-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← Σ_a π(a|s) Σ_s',r P(s', r|s, a) [r + γV(s')]",
    "Paradigm": "Dynamic Programming",
}

# Temporal Difference ---------------------------------------------------------------

algo_td0 = {
    "Algorithm": "TD(0)",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← V(s) + α[R_t+1 + γV(s') - V(s)]",
    "Paradigm": "Temporal Difference",
}

algo_sarsa = {
    "Algorithm": "SARSA",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Q",
    "Update Formula": "Q(s, a) ← Q(s, a) + α[R_t+1 + γQ(s', a') - Q(s, a)]",
    "Paradigm": "Temporal Difference",
}

algo_q_learning = {
    "Algorithm": "Q-Learning",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "Off-policy",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Q",
    "Update Formula": "Q(s, a) ← Q(s, a) + α[R_t+1 + γmax_a' Q(s', a') - Q(s, a)]",
    "Paradigm": "Temporal Difference",
}

algo_td_lambda = {
    "Algorithm": "TD(λ)",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "Something in between",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "V",
    "Update Formula": "V(s) ← V(s) + αδt Et[s]",
    "Paradigm": "Temporal Difference",
}

algo_double_q_learning = {
    "Algorithm": "Double Q-Learning",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "Off-policy",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Q",
    "Update Formula": "Q_1(s, a) ← Q_1(s, a) + α[R_t+1 + γQ_1(s', argmax_a' Q_2(s', a')) - Q_2(s, a)]",
    "Paradigm": "Temporal Difference",
}

algo_dynaq = {
    "Algorithm": "Dyna-Q",
    "Model-based vs Model-free": "Model-based",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "Shallow",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Q",
    "Update Formula": "Q(s, a) ← Q(s, a) + α[R_t+1 + γmax_a' Q(s', a') - Q(s, a)]",
    "Paradigm": "Temporal Difference",
}

# Policy Gradient ---------------------------------------------------------------

algo_reinforce = {
    "Algorithm": "REINFORCE",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "N/A",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "π",
    "Update Formula": "Δθ = α ∇θ log π(a|s) G_t",
    "Paradigm": "Policy Gradient",
}

# Actor-Critic ---------------------------------------------------------------

algo_q_actor_critic = {
    "Algorithm": "Q Actor-Critic",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "N/A",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "π, Q",
    "Update Formula": "Δθ = α ∇θ log π(a|s) Q(s,a)",
    "Paradigm": "Policy Gradient",
}

algo_advantage_actor_critic = {
    "Algorithm": "Advantage Actor-Critic",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "N/A",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "π, Q, V",
    "Update Formula": "Δθ = α ∇θ log π(a|s) A(s,a)",
    "Paradigm": "Policy Gradient",
}

algo_td_actor_critic = {
    "Algorithm": "TD Actor-Critic",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "N/A",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "V",
    "Update Formula": "Δθ = α ∇θ log π(a|s) δ",
    "Paradigm": "Policy Gradient",
}

algo_td_lambda_actor_critic = {
    "Algorithm": "TD(λ) Actor-Critic",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "N/A",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "V",
    "Update Formula": "Δθ = α ∇θ log π(a|s) δ_t E_t[s]",
    "Paradigm": "Policy Gradient",
}

algo_natural_actor_critic = {
    "Algorithm": "Natural Actor-Critic",
    "Model-based vs Model-free": "Model-free",
    "On-policy vs Off-policy": "On-policy",
    "Deep vs Shallow Update": "N/A",
    "Sample-based vs Tree-based Update": "Sample-based",
    "Target Function(s)": "Both",
    "Update Formula": "Δθ = α F^-1 ∇θ log π(a|s) δ",
    "Paradigm": "Policy Gradient",
}




# List of all algorithms
algos = [algo_mcc,
         algo_mcpe,
         algo_mcc_importance_sampling,
         algo_mcpe_importance_sampling,
         algo_iterative_policy_evaluation,
         algo_value_iteration,
         algo_policy_iteration,
         algo_td0,
         algo_sarsa,
         algo_q_learning,
         algo_double_q_learning]

# Create the DataFrame
df = pd.DataFrame(algos)
df = df.set_index("Algorithm")
st.dataframe(df, use_container_width=True)

st.markdown("""
# Notes & Observations
### General
* On-policy is a special case of off-policy, where the $\mu$ and $\pi$ are the same. So every on-policy algorithm can be used as off-policy algorithm, but not vice versa.
* If we target the Q function, we can use the algorithm for control, by choosing the action with the highest Q value. If we target the V function, we can use the algorithm for only for prediction.

### Monte Carlo
* Monte Carlo algorithms are model-free, because they don't need the transition probabilities and rewards.
* They do deep backups, they always see the whole episode, before they update the the Q or V function. This is why they they are inefficient, they need a only do one update per episode.
* They are sample-based, because they only use the samples from the environment.
* They don't use the markov property, so they work with non-markov environments.

### Dynamic Programming
* Dynamic Programming algorithms are model-based, because they follow the Bellman equation, which requires the transition probabilities and rewards.
* They do tree-based or full-backups, because they use the whole tree of possible states. And they can only do this, because they are model-based.

### Temporal Difference
* Temporal Difference algorithms are model-free, because they don't need the transition probabilities and rewards.
* But they implicitly learn a model of the MDP, because they use the Bellman equation. This is why they are considered biased.
* They update the Q or V function after every step, this is why they are more efficient than Monte Carlo algorithms.

""")