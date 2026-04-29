import pandas as pd
from typing import Any

# In-memory store for loaded datasets and MCMC traces.
# Both are keyed by user-supplied names.
datasets: dict[str, pd.DataFrame] = {}
mcmc_traces: dict[str, Any] = {}
