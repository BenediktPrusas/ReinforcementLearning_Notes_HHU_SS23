import streamlit as st
import streamlit.components.v1 as components

try:
    st.set_page_config(layout="wide", page_title="Reinforcement Learning Lecture Notes")
except:
    pass

def mermaid(code: str, height: int = 500) -> None:
    """Renders a mermaid diagram."""
    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """
    , height=height)

st.markdown("""
# Markov Models
On this page:
* [Markov Chain](#markov-chain)
* [Markov Reward Process](#markov-reward-process)
* [Markov Decision Process](#markov-decision-process)
""")

st.markdown("""
### Markov Chain
A Markov chain is a tuple $(S, P)$ where $S$ is a finite set of states and $P$ is a state transition probability matrix, $P_{ss'} = P[S_{t+1} = s' | S_t = s]$.
""")

mermaid(
    """
graph TD
    A(State A) -->|0.5| B(State B)
    A -->|0.5| C(State C)
    B -->|0.5| A
    B -->|0.5| C
    C -->|0.7| A
    C -->|0.3| C
    """
, height=280)

st.markdown("""
### Markov Reward Process
A Markov reward process is a tuple $(S, P, R, \gamma)$ where $S$ is a finite set of states, $P$ is a state transition probability matrix, $R$ is a reward function, $R_s = \mathbb{E}[R_{t+1} | S_t = s]$, and $\gamma$ is a discount factor, $\gamma \in [0, 1]$.
""")

mermaid(
    """
graph TD
    A[State A<br> Reward: 10] -->|0.5| B[State B<br> Reward: 20]
    A -->|0.5| C[State C<br> Reward: 30]
    B -->|0.5| A
    B -->|0.5| C
    C -->|0.7| A
    C -->|0.3| C
    """
, height=320)


st.markdown("""
### Markov Decision Process
A Markov decision process is a tuple $(S, A, P, R, \gamma)$ where $S$ is a finite set of states, $A$ is a finite set of actions, $P$ is a state transition probability matrix, $R$ is a reward function, $R_s = \mathbb{E}[R_{t+1} | S_t = s]$, and $\gamma$ is a discount factor, $\gamma \in [0, 1]$.
""")

mermaid(
    """
graph TB
    A[State A] -- Action X <br> Reward: 10 --> B[State B]
    A -- Action Y <br> Reward: 20 --> B
    B -- Action X <br> Reward: 10 --> A
    B -- Action Y <br> Reward: 20 --> A
    """
, height=280)

