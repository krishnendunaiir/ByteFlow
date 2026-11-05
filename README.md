# Autonomous AI Data Analyst

An agentic data-analysis assistant that receives structured datasets (CSV or Excel) and natural-language queries to execute, evaluate, and report statistical and machine-learning workflows under explicit programmatic constraints.

---

> **Key Architectural Takeaway:** The AI **plans and interprets**, while deterministic, tested Python code executes every calculation, ensures statistical validity, and validates every generated metric before report assembly.

---

## 1. Overview & Problem Statement

Exploratory analysis and data modeling require multiple manual, error-prone steps: quality auditing, statistical test selection, assumption validation, leakage-free modeling, visualization, and technical reporting. Standard "chat with your data" solutions often execute arbitrary code or rely on LLM-driven arithmetic—introducing risks like hallucinated metrics, unhandled data leakage, code injection, and prompt manipulation.

This project addresses those failure modes by separating reasoning from computation:
- **Zero dynamic code execution:** The LLM selects actions from a strictly controlled schema; deterministic functions perform all computations.
- **Fact-checked outputs:** Every number appearing in the generated narrative is verified against pre-computed tool outputs.
- **Leak-proof ML pipelines:** Data transformation, baseline comparisons, cross-validation, and holdout evaluations follow strict scikit-learn pipeline standards.

---

## 2. Core Architecture & Workflow

The system is organized as an orchestrated state graph managed via **LangGraph**. Workflows proceed through distinct state nodes, incorporating rule-based fallbacks and automated critic checks.

```mermaid
flowchart TD
    S([START]) --> V[Validate Node]
    V --> U[Understand Node<br/>profile + objective]
    U --> Q[Quality Node<br/>rule-based checks]
    Q --> P[Plan Node<br/>AI proposes, code validates]
    P --> A[Analyse Node<br/>statistics tools]
    A -->|plan needs ML| M[Machine Learning Node]
    A -->|no ML needed| Z[Visualize Node]
    M --> Z
    Z --> C[Critic Node]
    C -->|approved, or retry limit reached| R[Report Node] --> E([END])
    C -.->|blocking issue, retries left| P
    C -.->|blocking issue, retries left| A
    C -.->|blocking issue, retries left| Z
```

### Agent Roles & Capabilities

| Agent / Node | Primary Responsibility | LLM Driven? | Security & Validation Controls |
| :--- | :--- | :---: | :--- |
| **Validate** | Assesses file structure, bounds, and query validity | **No** | Rejects malformed requests prior to LLM invocation. |
| **Understand** | Summarizes schema, column types, and objectives | **Yes** | Receives only sanitized column summaries (no raw data rows). |
| **Quality** | Identifies missingness, duplicates, outliers, and data type anomalies | **No** | Uses deterministic rules (e.g., IQR for outliers, placeholder checks). |
| **Plan** | Proposes statistical tests and modeling strategies | **Yes** | Proposals are strictly validated against dataset metadata and drop invalid actions. |
| **Analyse** | Executes requested statistical tests | **No** | Runs hypothesis testing with assumption checks and effect sizes. |
| **Machine Learning**| Trains baselines, candidate models, and evaluates feature importance | **No** | Enforces leak-safe transformations, cross-validation, and holdout evaluation. |
| **Visualize** | Generates relevant static charts | **No** | Constrained to deterministic Matplotlib/Seaborn plot generation. |
| **Critic** | Evaluates analytical completeness and structural integrity | **Partial** | Enforces execution limits (`MAX_RETRIES`) and verifies output sanity. |
| **Report** | Compiles the final 11-section structured analytical report | **Yes** | Narrative text is validated against numerical tool outputs to eliminate hallucinated stats. |

---

## 3. Key Features

- **Automated Data Quality Auditing:** Detects missing entries, duplicate rows, numerical outliers (IQR), type mismatches, inconsistent categoricals, and sentinel placeholder values (e.g., `999`).
- **Context-Aware Hypothesis Testing:** Automatically selects appropriate parametric or non-parametric tests based on data distribution and sample sizes:
  - *Correlation:* Pearson, Spearman (with multiple-testing corrections).
  - *Group Comparisons:* Welch's t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis, Chi-Square, Fisher's Exact test.
  - *Outputs:* Test statistics, p-values, effect sizes, and explicit reasoning.
- **Leak-Safe Machine Learning:**
  - Automatic task routing (Classification, Regression, or Clustering).
  - Baseline comparison, cross-validation, and final holdout set evaluation.
  - Permutation feature importance and automated overfitting/weak-model warnings.
- **Fail-Safe Resilience:** Includes rule-based plan fallbacks if the LLM provider experiences downtime or API limits.

---

## 4. Technical Stack

- **Orchestration & Workflow:** Python 3.12+, LangGraph, Pydantic
- **Data & Computation:** Pandas, NumPy, SciPy, statsmodels, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **User Interface:** Streamlit
- **LLM Integration:** OpenAI SDK (Compatible with Groq, Google Gemini, OpenRouter, and OpenAI)

---

## 5. System Specifications & Security Overview

### File & Data Handling Limits
- **Supported Formats:** `.csv`, `.xlsx` (first worksheet).
- **Dataset Capacity:** Up to 200,000 rows and 200 columns.
- **Execution Storage:** In-memory dataset handling; auto-cleared temporary artifacts.

### Security Highlights
- **No Unsafe Code Execution:** The LLM does not write or execute Python scripts.
- **Privacy Controls:** Raw dataset rows are never exposed to external LLM endpoints—only aggregated metadata and column signatures are transmitted.
- **Injection Mitigation:** Dataset headers and content strings are sanitized before display or prompt insertion; Excel formula evaluation is disabled.

---

## 6. Known Scope & Limitations

- **Analytical Boundaries:** Focuses on tabular classification, regression, clustering, and fundamental bivariate/multivariate statistics. Does not cover time-series forecasting or complex NLP pipelines.
- **Observational Limitations:** Outputs describe statistical associations and predictive patterns rather than causal relationships.