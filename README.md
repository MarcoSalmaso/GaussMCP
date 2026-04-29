# GaussMCP

**GaussMCP** è un server MCP (Model Context Protocol) in Python che espone **25 tool di analisi statistica** direttamente nei tuoi LLM preferiti. Permette di eseguire analisi descrittive, test inferenziali, modelli time series e inferenza bayesiana su dati provenienti da file CSV/Excel, database PostgreSQL o passati inline come JSON.

---

## Indice

- [Installazione](#installazione)
- [Configurazione MCP](#configurazione-mcp)
- [Tool disponibili](#tool-disponibili)
  - [Caricamento dati](#-caricamento-dati-7-tool)
  - [Statistiche descrittive](#-statistiche-descrittive-4-tool)
  - [Statistica inferenziale](#-statistica-inferenziale-5-tool)
  - [Analisi time series](#-analisi-time-series-5-tool)
  - [Analisi bayesiana](#-analisi-bayesiana-4-tool)
- [Background statistico](#background-statistico)
- [Esempi di utilizzo](#esempi-di-utilizzo)

---

## Installazione

### Requisiti

- Python 3.11+
- pip

### Installazione base

```bash
git clone https://github.com/MarcoSalmaso/GaussMCP.git
cd GaussMCP
pip install -e .
```

### Installazione con supporto Bayesiano (PyMC)

I tool MCMC richiedono PyMC, che è una dipendenza opzionale (pesante, ~500MB):

```bash
pip install -e ".[bayesian]"
```

### Dipendenze incluse

| Pacchetto | Utilizzo |
|---|---|
| `mcp` | SDK ufficiale MCP di Anthropic |
| `pandas` | Manipolazione dati |
| `numpy` | Calcolo numerico |
| `scipy` | Test statistici |
| `statsmodels` | Regressione, ARIMA, time series |
| `openpyxl` | Lettura file Excel |
| `psycopg2-binary` | Driver PostgreSQL |
| `sqlalchemy` | Connessione database |
| `pymc` *(opzionale)* | MCMC e inferenza bayesiana |
| `arviz` *(opzionale)* | Diagnostica MCMC |

---

## Configurazione MCP

### Claude Code (CLI)

```bash
# Scope locale (solo progetto corrente)
claude mcp add GaussMCP /path/to/gauss-mcp

# Scope utente (disponibile ovunque)
claude mcp add --scope user GaussMCP /path/to/gauss-mcp
```

### claude_desktop_config.json / .mcp.json / config generico

```json
{
  "mcpServers": {
    "GaussMCP": {
      "type": "stdio",
      "command": "/Library/Frameworks/Python.framework/Versions/3.13/bin/gauss-mcp",
      "args": []
    }
  }
}
```

> **Nota:** sostituisci il `command` con il path corretto sul tuo sistema.
> Puoi trovarlo con: `which gauss-mcp`

### LLM locale su macchina diversa o container

```bash
# Installa il pacchetto nell'ambiente target
pip install -e /path/to/GaussMCP

# Usa questo command nel config
"command": "python3"
"args": ["-m", "gauss_mcp.server"]
```

---

## Tool disponibili

### 📂 Caricamento dati (7 tool)

I dati vengono caricati in memoria con un nome assegnato dall'utente e rimangono disponibili per tutta la sessione. Più tool possono operare sullo stesso dataset senza ricaricare i dati.

---

#### `load_csv`
Carica un file CSV in memoria come dataset nominato.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `file_path` | str | — | Percorso assoluto o relativo al file |
| `dataset_name` | str | — | Nome da assegnare al dataset |
| `separator` | str | `,` | Delimitatore di colonna |
| `encoding` | str | `utf-8` | Encoding del file |
| `parse_dates` | bool | `True` | Tenta il parsing automatico delle date |

---

#### `load_excel`
Carica un file Excel (.xlsx, .xls) in memoria.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `file_path` | str | — | Percorso al file Excel |
| `dataset_name` | str | — | Nome da assegnare al dataset |
| `sheet_name` | str | `"0"` | Nome o indice del foglio |
| `parse_dates` | bool | `True` | Tenta il parsing automatico delle date |

---

#### `load_from_sql`
Esegue una query SQL su PostgreSQL e carica il risultato come dataset.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `connection_string` | str | — | Stringa di connessione PostgreSQL |
| `query` | str | — | Query SELECT da eseguire |
| `dataset_name` | str | — | Nome da assegnare al dataset |

**Formato connection string:**
```
postgresql://utente:password@host:5432/nome_database
```

---

#### `load_from_json`
Carica dati passati inline come stringa JSON.

Accetta due formati:
- **Records:** `[{"col1": val, "col2": val}, ...]`
- **Colonne:** `{"col1": [val1, val2], "col2": [val1, val2]}`

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `data` | str | — | Stringa JSON con i dati |
| `dataset_name` | str | — | Nome da assegnare al dataset |

---

#### `list_datasets`
Elenca tutti i dataset attualmente in memoria con shape e prime colonne.

---

#### `preview_dataset`
Mostra le prime N righe di un dataset.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `rows` | int | `10` | Numero di righe da mostrare |

---

#### `drop_dataset`
Rimuove un dataset dalla memoria.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset da rimuovere |

---

### 📊 Statistiche descrittive (4 tool)

---

#### `describe`
Calcola le statistiche descrittive complete per le colonne numeriche di un dataset.

**Output:**
- Count, mean, std, min, max
- Percentili: 5°, 25°, 50°, 75°, 95°
- Skewness (asimmetria) e Kurtosis (curtosi)
- Valori mancanti (count e %)

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `columns` | list[str] | `None` | Colonne da includere (default: tutte le numeriche) |

---

#### `frequency_table`
Calcola una tabella di frequenza per una colonna.

- **Colonne categoriali/stringa:** conta le occorrenze di ogni valore unico
- **Colonne numeriche con `bins`:** raggruppa i valori in intervalli

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `column` | str | — | Colonna da analizzare |
| `bins` | int | `None` | Numero di classi per dati numerici |
| `normalize` | bool | `False` | Restituisce proporzioni invece di conteggi |

---

#### `correlation_matrix`
Calcola la matrice di correlazione tra colonne numeriche.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `columns` | list[str] | `None` | Colonne da includere |
| `method` | str | `pearson` | Metodo: `pearson`, `spearman`, `kendall` |

**Quando usare quale metodo:**
- **Pearson:** relazioni lineari tra variabili continue normalmente distribuite
- **Spearman:** relazioni monotone, robusto agli outlier, per dati ordinali
- **Kendall:** correlazione di rango, più robusto con campioni piccoli

---

#### `distribution_fit`
Fitta una o più distribuzioni di probabilità a una colonna numerica e le classifica per bontà di adattamento.

**Distribuzioni disponibili (default):** `norm`, `expon`, `gamma`, `lognorm`, `beta`, `weibull_min`

**Output per ogni distribuzione:**
- Statistica KS (Kolmogorov-Smirnov) e p-value
- AIC (Akaike Information Criterion)
- Parametri stimati

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `column` | str | — | Colonna numerica da analizzare |
| `distributions` | list[str] | `None` | Lista di distribuzioni scipy.stats da testare |

---

### 🔬 Statistica inferenziale (5 tool)

---

#### `t_test`
Esegue un t-test su una colonna numerica.

**Tipologie disponibili:**

| `test_type` | Descrizione | Parametri aggiuntivi |
|---|---|---|
| `one_sample` | Testa se la media della colonna è uguale a `popmean` | `popmean`, `alternative` |
| `two_sample` | Confronta le medie di due gruppi | `group_column`, `alternative` |
| `paired` | T-test su coppie di misurazioni | `group_column` (seconda colonna), `alternative` |

**Note:**
- Il two-sample test applica automaticamente la **correzione di Welch** se il test di Levene rifiuta l'omogeneità delle varianze (p < 0.05)
- `alternative`: `two-sided` (default), `less`, `greater`
- Output include: t, df, p-value, intervallo di confidenza al 95%

---

#### `anova`
Esegue una One-Way ANOVA per confrontare le medie tra più gruppi.

**Output:**
- Statistiche descrittive per gruppo (n, media, std)
- F-statistic e p-value
- Post-hoc Tukey's HSD (se p < 0.05 e `post_hoc=True`)

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `value_column` | str | — | Colonna con i valori misurati |
| `group_column` | str | — | Colonna con le etichette di gruppo |
| `post_hoc` | bool | `True` | Esegue Tukey HSD se ANOVA è significativa |

---

#### `chi_square_test`
Test chi-quadro di indipendenza tra due variabili categoriali.

**Output:**
- Tabella di contingenza
- χ², gradi di libertà, p-value
- **Cramér's V** (misura dell'effetto): piccolo < 0.1, medio < 0.3, grande ≥ 0.3
- Avviso se frequenze attese < 5

---

#### `linear_regression`
Fitta un modello di regressione lineare OLS e restituisce una diagnostica completa.

**Output:**
- Tabella coefficienti: coef, std_err, t, p-value, CI 95%
- R², R² aggiustato
- F-statistic e p-value del modello
- AIC, BIC
- Test di normalità dei residui (Shapiro-Wilk)

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `target` | str | — | Variabile dipendente |
| `features` | list[str] | — | Variabili indipendenti |
| `intercept` | bool | `True` | Includi il termine costante |

---

#### `normality_test`
Testa se una colonna numerica segue una distribuzione normale.

| Test | Descrizione | Consigliato per |
|---|---|---|
| `shapiro` | Shapiro-Wilk | n ≤ 5000 (più potente) |
| `ks` | Kolmogorov-Smirnov | Qualsiasi n |
| `dagostino` | D'Agostino-Pearson | n > 20 (basato su skewness e kurtosis) |
| `all` | Tutti e tre | Default |

**H₀ (ipotesi nulla):** i dati seguono una distribuzione normale.
Un p-value > 0.05 non rifiuta H₀ → i dati *appaiono* normali.

---

### 📈 Analisi time series (5 tool)

---

#### `decompose_time_series`
Decompone una serie temporale nelle sue componenti fondamentali.

**Componenti:**
- **Trend:** andamento di lungo periodo
- **Stagionalità:** pattern ciclico ripetuto
- **Residuo:** variazione non spiegata

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `value_column` | str | — | Colonna con i valori |
| `date_column` | str | `None` | Colonna data/datetime per ordinamento |
| `model` | str | `additive` | `additive` o `multiplicative` |
| `period` | int | `None` | Periodo stagionale (es. 12 per mensile, 7 per settimanale) |

**Quando usare il modello moltiplicativo:** quando l'ampiezza della stagionalità cresce proporzionalmente al trend (es. vendite in crescita con picchi natalizi sempre più alti).

---

#### `stationarity_test`
Testa la stazionarietà di una serie temporale.

| Test | H₀ | Rifiuto H₀ |
|---|---|---|
| **ADF** (Augmented Dickey-Fuller) | Presenza di radice unitaria (non stazionaria) | Serie stazionaria |
| **KPSS** | Serie stazionaria | Serie non stazionaria |

> Usarli insieme è consigliato: ADF e KPSS concordi danno maggiore certezza.

---

#### `autocorrelation`
Calcola ACF (Autocorrelation Function) e PACF (Partial ACF).

**Interpretazione per la scelta dell'ordine ARIMA:**
| Pattern | Suggerisce |
|---|---|
| ACF decade lentamente | Serie non stazionaria → differenziare (d > 0) |
| ACF troncata al lag q | Componente MA(q) |
| PACF troncata al lag p | Componente AR(p) |
| Entrambe decadono | Modello ARMA misto |

---

#### `fit_arima`
Fitta un modello ARIMA o SARIMA a una serie temporale.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `dataset_name` | str | — | Nome del dataset |
| `column` | str | — | Colonna della serie temporale |
| `order` | list[int] | `[1,1,1]` | Ordine (p, d, q) |
| `seasonal_order` | list[int] | `None` | Ordine stagionale (P, D, Q, s), es. `[1,1,1,12]` |
| `date_column` | str | `None` | Colonna data per ordinamento |

**Output:** coefficienti, AIC/BIC/HQIC, log-likelihood, diagnostica residui.

---

#### `forecast_arima`
Fitta un ARIMA e genera previsioni multi-step con intervalli di confidenza.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `steps` | int | `10` | Periodi futuri da prevedere |
| `confidence` | float | `0.95` | Livello dell'intervallo di confidenza |

---

### 🎲 Analisi bayesiana (4 tool)

---

#### `bayesian_update`
Aggiornamento bayesiano in forma chiusa usando prior coniugati.

**Coppie prior-likelihood supportate:**

| Prior | Likelihood | Stima | `prior_params` JSON |
|---|---|---|---|
| **Beta** | **Binomiale** | Proporzione p | `{"alpha": 1, "beta": 1}` |
| **Normale** | **Normale** | Media μ (varianza nota) | `{"mu": 0, "sigma": 1, "likelihood_sigma": 1}` |
| **Gamma** | **Poisson** | Tasso λ | `{"alpha": 1, "beta": 1}` |

**Output:** parametri del posterior, media, moda, std, intervallo di credibilità al 95% (CrI).

**Differenza tra CI frequentista e CrI bayesiano:**
- **CI 95%:** in 95 esperimenti su 100, l'intervallo contiene il vero parametro
- **CrI 95%:** dato i dati osservati, c'è il 95% di probabilità che il parametro sia nell'intervallo

---

#### `mcmc_sample`
Campionamento MCMC con PyMC per inferenza bayesiana completa.

> Richiede: `pip install "gauss-mcp[bayesian]"`

**Modelli disponibili:**

| `model_type` | Descrizione | Parametri richiesti |
|---|---|---|
| `normal_mean` | Stima μ e σ di una popolazione normale | `column` |
| `proportion` | Stima proporzione da dati binari (0/1) | `column` |
| `linear_regression` | Regressione OLS bayesiana con prior debolmente informativi | `target`, `features` |

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `draws` | int | `1000` | Campioni posterior per catena |
| `tune` | int | `1000` | Step di warm-up per catena |
| `chains` | int | `2` | Numero di catene indipendenti |
| `trace_name` | str | `"default"` | Nome per salvare la trace |

---

#### `posterior_summary`
Mostra il sommario del posterior e le diagnostiche di convergenza per una trace MCMC.

**Diagnostiche:**
- **R̂ (R-hat):** misura di convergenza. R̂ < 1.01 = ottima convergenza; > 1.05 = problemi
- **ESS (Effective Sample Size):** numero effettivo di campioni indipendenti. ESS > 400 è accettabile

---

#### `list_traces`
Elenca tutte le trace MCMC attualmente in memoria con variabili e dimensioni.

---

## Background statistico

### Concetti chiave

**p-value:** probabilità di osservare un risultato almeno altrettanto estremo, *assumendo vera l'ipotesi nulla*. Non è la probabilità che H₀ sia vera.

**Livello α (alpha):** soglia di significatività. Per convenzione α = 0.05 (5%).
- p < α → si rifiuta H₀
- p ≥ α → non si rifiuta H₀ (non significa che H₀ è vera)

**Potenza di un test:** probabilità di rifiutare H₀ quando è effettivamente falsa. Dipende da n, effect size e α.

**Assunzioni dei test parametrici:**
- T-test, ANOVA, regressione OLS → normalità dei residui, omoschedasticità
- Se violate: usare alternative non parametriche (Mann-Whitney, Kruskal-Wallis, Spearman)

### Scelta del test corretto

```
Confronto di medie?
├── 2 gruppi, dati appaiati → t_test (paired)
├── 2 gruppi, indipendenti  → t_test (two_sample)
└── ≥3 gruppi               → anova

Associazione tra variabili?
├── Entrambe numeriche       → correlation_matrix
├── Una numerica, una categ. → anova
└── Entrambe categoriali     → chi_square_test

Predizione?
└── Target numerico          → linear_regression

Distribuzione di una variabile?
├── È normale?               → normality_test
└── Che distribuzione segue? → distribution_fit
```

---

## Esempi di utilizzo

### 1. Analisi esplorativa da CSV

```
"Carica il file /dati/clienti.csv come 'clienti',
 fammi una panoramica statistica completa e
 controlla se l'età è normalmente distribuita"
```

Il modello userà: `load_csv` → `describe` → `normality_test`

---

### 2. Confronto tra gruppi da database

```
"Carica la query:
   SELECT prodotto, vendite, regione FROM ordini WHERE anno = 2024
 dal database postgresql://user:pass@localhost/mydb come 'ordini2024'.
 Confronta le vendite tra le regioni con un test appropriato."
```

Il modello userà: `load_from_sql` → `normality_test` → `anova` (con Tukey HSD)

---

### 3. Previsione time series

```
"Ho i dati mensili di fatturato in /dati/fatturato.csv.
 Caricali come 'fatturato', verifica la stazionarietà,
 analizza ACF e PACF, poi fitta un ARIMA e prevedi i prossimi 12 mesi."
```

Il modello userà: `load_csv` → `stationarity_test` → `autocorrelation` → `fit_arima` → `forecast_arima`

---

### 4. Aggiornamento bayesiano di una proporzione

```
"Ho un prior Beta(2,5) su una proporzione.
 Ho osservato 15 successi su 40 prove.
 Aggiorna il posterior e dimmi la stima con intervallo di credibilità."
```

Il modello userà: `bayesian_update` con `prior_type="beta"`, `prior_params='{"alpha":2,"beta":5}'`

---

### 5. Regressione OLS con diagnostica

```
"Carica questi dati:
 [{'prezzo': 250, 'mq': 80, 'zona': 1}, {'prezzo': 310, 'mq': 95, 'zona': 2}, ...]
 come 'immobili' e analizza quanto mq e zona spiegano il prezzo."
```

Il modello userà: `load_from_json` → `correlation_matrix` → `linear_regression`

---

### 6. MCMC Bayesiano completo

```
"Carica i dati di conversione da /dati/campagna.csv come 'campagna'
 (colonna 'convertito' è 0/1).
 Stima la probabilità di conversione con MCMC bayesiano,
 usa 2000 draws e mostrami le diagnostiche di convergenza."
```

Il modello userà: `load_csv` → `mcmc_sample` (proportion) → `posterior_summary`

---

## Struttura del progetto

```
GaussMCP/
├── src/gauss_mcp/
│   ├── app.py              # Istanza FastMCP
│   ├── state.py            # Stato in memoria (datasets, traces)
│   ├── server.py           # Entry point
│   ├── tools/
│   │   ├── data_loader.py  # Caricamento dati
│   │   ├── descriptive.py  # Statistiche descrittive
│   │   ├── inferential.py  # Test inferenziali
│   │   ├── time_series.py  # Analisi temporali
│   │   └── bayesian.py     # Inferenza bayesiana
│   └── utils/
│       └── formatters.py   # Utilities di formattazione
└── pyproject.toml
```

---

## Licenza

MIT
