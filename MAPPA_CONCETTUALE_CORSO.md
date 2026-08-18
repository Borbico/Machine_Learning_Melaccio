# Mappa Concettuale, Compendio Formule e Guida al Codice
**Corso del Prof. Alessio Micheli — Università di Pisa**

---

## 🗺️ Mappa Concettuale Globale

```mermaid
flowchart TD
    ML["MACHINE LEARNING"] --> P1["1. IL FRAMEWORK FONDAMENTALE\n(I 4 Elementi di ogni algoritmo)"]
    ML --> P2["2. APPRENDIMENTO SUPERVISIONATO\n(Modelli Lineari, NNs, SVM, k-NN)"]
    ML --> P3["3. VALIDAZIONE & STATISTICAL LEARNING THEORY\n(Bias-Varianza, SLT, VC-Dim, SRM)"]
    ML --> P4["4. ARCHITETTURE AVANZATE & UN-SUPERVISED\n(Autoencoder, RNN, SOM, GNN, Random NN)"]
    ML --> P5["5. GUIDA INTEGRALE AL CODICE DEL PROGETTO\n(File .py e Notebook .ipynb)"]

    P1 --> F1["Dati X, y"]
    P1 --> F2["Spazio delle Ipotesi H"]
    P1 --> F3["Loss Function L"]
    P1 --> F4["Algoritmo di Ottimizzazione"]

    P2 --> M1["Modelli Lineari (Perceptron, LMS, Logistic)"]
    P2 --> M2["Reti Neurali Multi-layer (MLP & Backprop)"]
    P2 --> M3["Support Vector Machines (SVM & SVR)"]
    P2 --> M4["Instance-Based (k-NN)"]

    P3 --> V1["Bias-Variance-Noise Decomposition"]
    P3 --> V2["Schemi CV & Prevenzione Data Leakage"]
    P3 --> V3["VC-Dimension & Generalization Bounds"]
    P3 --> V4["Structural Risk Minimization (SRM)"]

    P4 --> A1["Deep Learning (CNN, RNN, Autoencoder)"]
    P4 --> A2["Algoritmi Costruttivi (Cascade Correlation)"]
    P4 --> A3["Random NN & Reservoir Computing"]
    P4 --> A4["SOM & Graph Neural Networks (Message Passing)"]

    P5 --> C1["File Python (.py): cross_common, nn_common, ecc."]
    P5 --> C2["Notebook MONK: monk_KNN, monk_SVM, monk_NN"]
    P5 --> C3["Notebook CUP: cup_KNN, cup_SVM, cup_NN, cup_Ensemble"]
```

---

# 📌 SEZIONE 1: Il Framework Fondamentale & Modelli Lineari

### 1. Il Framework dei 4 Elementi
* **Dati**: $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$ dove $x_i \in \mathbb{R}^D$ e $y_i \in \mathcal{Y}$ (coppie input-target).
* **Spazio Ipotesi $\mathcal{H}$**: L'insieme di tutte le funzioni rappresentabili dal modello.
* **Loss Function $\mathcal{L}$**:
  * **MSE Loss**: $L_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ *(Nota di Micheli: N è il numero di coppie dato-target!)*
  * **MEE Loss (ML CUP)**: $L_{MEE} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2 = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^K (y_{i,m} - \hat{y}_{i,m})^2}$
  * **BCE Loss**: $L_{BCE} = -\frac{1}{N} \sum_{i=1}^N \big[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \big]$
* **Ottimizzazione**: Discesa del gradiente $w^{(t+1)} = w^{(t)} - \eta \nabla L(w)$.

---

### 2. Perceptron (LTU) & Teorema di Convergenza
* **Funzione di Attivazione a Gradino**: $f(x) = \text{sign}(w^T x + b)$.
* **Regola di Aggiornamento Pesi**: $w \leftarrow w + \eta (y_i - o_i) x_i$.
* **Teorema di Convergenza del Perceptron**: Se un dataset è linearmente separabile con un margine $\gamma > 0$ (ovvero esiste un peso $w^*$ tale che $y_i (w^{*T} x_i) \ge \gamma$), allora l'algoritmo convergerà commettendo al massimo un numero finito di errori:
  $$k \le \frac{R^2}{\gamma^2}$$
  dove $R = \max_i \|x_i\|$ è il raggio del dataset.

#### 💡 Domanda Frequente: Creare la porta NOT con 1 Perceptrone
* Input $x \in \{0, 1\}$, Target $y = 1 - x$.
* Pesi: $w = -2$, bias $b = 1$.
* Output: $f(x) = \text{sign}(-2x + 1)$. Se $x=0 \rightarrow \text{sign}(1)=+1$. Se $x=1 \rightarrow \text{sign}(-1)=-1$.

---

### 3. Funzioni di Attivazione e loro Proprietà
* **Sigmoide**: $\sigma(z) = \frac{1}{1 + e^{-z}}$. Derivata: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
  * *Perché si usa invece del gradino?* È continua e **derivabile** (essenziale per la Backpropagation).
  * *Svantaggi*: Satura per $|z|$ grandi, causando il fenomeno del *Vanishing Gradient*.
* **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$.
  * Derivata: $f'(x) = 1$ se $x > 0$, $0$ se $x \le 0$.
  * *Significato della derivata (Domanda d'Esame)*: Agisce come una **maschera binaria (0/1)** basata sul valore di $net$, stabilendo quali neuroni trasmettono il flusso del gradiente all'indietro.
  * *Perché aiuta per il Vanishing Gradient?* Per $x > 0$ la derivata è esattamente 1, permettendo al gradiente di fluire inalterato senza degradarsi.
  * *Svantaggi*: "Dying ReLU" (se un neurone va sotto zero, la derivata vale 0 ed il peso non si aggiorna più). Alternative: LeakyReLU $f(x) = \max(\alpha x, x)$, GELU, ELU.
* **Perché NON usare solo attivazioni lineari?**
  * Se usassimo funzioni lineari $f(x) = c \cdot x$, la composizione di più strati collasserebbe algebricamente in un'unica trasformazione lineare finale $W_{tot} x$, perdendo tutta la capacità espressiva non lineare.

---

# 📌 SEZIONE 2: Reti Neurali & Backpropagation

### 1. Iperparametri dell'Aggiornamento dei Pesi
$$\Delta w(t) = -\eta \frac{\partial E}{\partial w(t)} + \alpha \Delta w(t-1) - \eta \lambda w(t)$$
* **$\eta$ (Learning Rate)**: Controlla la lunghezza del passo di discesa.
* **$\alpha$ (Momentum / Inerzia)**: Mantiene una frazione dell'aggiornamento dell'epoca precedente per smorzare le oscillazioni ed evitare minimi locali poco profondi.
* **$\lambda$ (Weight Decay / Regolarizzazione $L_2$)**: Aggiunge una sanzione $\frac{\lambda}{2} \|w\|^2$ alla loss, riducendo progressivamente la norma dei pesi per evitare l'overfitting.

---

### 2. Dimostrazione della Backpropagation (Derivazione dei $\delta$)
Vogliamo calcolare $\Delta_p w_{tu} = -\frac{\partial E_p}{\partial w_{tu}}$.

Usando la Chain Rule:
$$\Delta_p w_{tu} = -\frac{\partial E_p}{\partial net_t} \cdot \frac{\partial net_t}{\partial w_{tu}} = \delta_t \cdot o_u$$

I due casi per il calcolo di $\delta_t$:
* **Neurone di Output ($t = k$)**:
  $$\delta_k = -\frac{\partial E_p}{\partial o_k} \frac{\partial o_k}{\partial net_k} = (d_k - o_k) f'_k(net_k)$$
* **Neurone Nascosto ($t = j$)**:
  $$\delta_j = \left( \sum_{k} \delta_k w_{kj} \right) f'_j(net_j)$$
* **Fattorizzazione ed Efficienza**: Calcolare un solo $\delta_t$ per neurone riduce il costo computazionale da $O(|W|^2)$ a $O(|W|)$ (lineare nel numero di pesi!).

---

# 📌 SEZIONE 3: Support Vector Machines (SVM & SVR)

### 1. Hard Margin SVM (Margine Rigido)
* **Margine Geometrico**: $M = \frac{2}{\|w\|}$.
* **Formulazione Primate**:
  $$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{sotto i vincoli } y_i(w^T x_i + b) \ge 1 \quad \forall i$$

---

### 2. Soft Margin SVM ($C$ e Slack Variables $\xi_i$)
* **Formulazione Primate**:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{sotto i vincoli } y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
* **Ruolo di $C$ (Domanda da Scritto/Orale)**:
  * $C \to \infty$: Penalizzazione massima degli errori $\rightarrow$ Margine stretto $\rightarrow$ **Rischio Overfitting**.
  * $C$ piccolo: Maggiore tolleranza per le violazioni $\rightarrow$ Margine ampio $\rightarrow$ **Regolarizzazione (Underfitting se troppo piccolo)**.

---

### 3. Formulazione Duale & Kernel Trick
* **Formulazione Duale**:
  $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{sotto } 0 \le \alpha_i \le C, \,\, \sum \alpha_i y_i = 0$$
* **Sparsità KKT**: Solo i punti sul margine hanno $\alpha_i > 0$ (**Vettori di Supporto**).
* **Kernel RBF (Gaussiano)**:
  $$K(x, z) = \exp(-\gamma \|x - z\|^2)$$
  * $\gamma$ alto $\rightarrow$ raggio di influenza piccolo $\rightarrow$ **Overfitting**.
  * $\gamma$ basso $\rightarrow$ raggio di influenza ampio $\rightarrow$ **Underfitting**.

---

### 4. Support Vector Regression (SVR ed la $\epsilon$-Tube)
* **$\epsilon$-Insensitive Loss**:
  $$|y - f(x)|_\epsilon = \max(0, |y - f(x)| - \epsilon)$$
* L'errore è pari a zero se il punto rientra nel tubo di tolleranza $\pm \epsilon$ attorno alla funzione predetta.

---

# 📌 SEZIONE 4: Statistical Learning Theory (SLT) & Validazione

### 1. Dimostrazione della Decomposizione Bias-Varianza-Rumore
Dato $y = f(x) + \epsilon$ con $\mathbb{E}[\epsilon]=0$ e $\mathbb{E}[\epsilon^2]=\sigma^2$:
$$\mathbb{E}_{D, \epsilon} \big[ (y - h_D(x))^2 \big] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D [ (h_D(x) - \bar{h}(x))^2 ]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$

---

### 2. VC-Dimension (Vapnik-Chervonenkis)
* **Shattering (Frammentazione)**: Un insieme di $N$ punti è shatterato da $\mathcal{H}$ se il modello può realizzare tutte le $2^N$ dicotomie (+1 / -1).
* **Definizione**: $VC(\mathcal{H})$ è la **massima cardinalità $N$** di punti shatterabili da $\mathcal{H}$.
* **VC-Dimension nei vari modelli**:
  * Iperpiani in $\mathbb{R}^D$ (Perceptron): $VC = D + 1$.
  * Reti Neurali: Cresce col numero di pesi $O(|W| \log |W|)$.
  * **SVM**: $VC \le \min\left(D, \frac{R^2}{\gamma^2}\right) + 1$. **NON dipende dalla dimensione $D$ dell'input**, ma dipende inversamente dal margine $\gamma$! (Ecco perché l'SVM evita la *curse of dimensionality*).
  * **1-NN**: $VC = \infty$ (memorizza qualsiasi dataset).

---

### 3. Structural Risk Minimization (SRM)
$$R(h) \le R_{emp}(h) + \Omega\left(\frac{h_{VC}}{N}\right)$$
* SRM organizza le ipotesi in strutture annidate $\mathcal{H}_1 \subset \mathcal{H}_2 \dots \subset \mathcal{H}_k$ e sceglie l'ipotesi che minimizza la somma dell'errore empirico e della confidenza VC.
* **Early Stopping come Regolarizzazione**: Limitando il numero di epoche, si limita implicitamente la norma dei pesi e la dimensione VC della rete!

---

# 📌 SEZIONE 5: Architetture Avanzate (Domande Scritte e Orali 2026)

### 1. Cascade Correlation (Algoritmo Costruttivo)
* **Principio**: Parte da una rete senza hidden unit e aggiunge un neurone alla volta **in cascata** (collegato all'input e a tutti i neuroni nascosti precedenti).
* **Funzione Obiettivo del Candidato**: Massimizzare la covarianza $S$ con l'errore residuo:
  $$S = \sum_k \left| \sum_p (o_p - \bar{o})(E_{p,k} - \bar{E}_k) \right|$$
* **Aggiornamento pesi del candidato**: Ascesa del gradiente $\Delta w_j = +\eta \frac{\partial S}{\partial w_j}$.
* **Congelamento (Frozen Weights)**: Una volta inserito il candidato, i suoi pesi d'ingresso vengono **congelati per sempre**, e si ri-addestrerà solo il layer di output.

---

### 2. Autoencoders (Undercomplete vs Overcomplete)
* **Undercomplete Autoencoder**: La dimensione dello strato nascosto $z$ (bottleneck) è **inferiore** all'input ($dim(z) < dim(x)$). Forzano la rete ad apprendere una rappresentazione latente compressa (simile alla PCA non lineare).
* **Overcomplete Autoencoder**: La dimensione dello strato nascosto è **superiore** all'input ($dim(z) > dim(x)$). Usati con regolarizzazione (es. Sparsity o Denoising) per apprendere feature ricche e ridondanti senza memorizzare l'input.

---

### 3. Random Neural Networks & Reservoir Computing (Echo State Networks)
* **Principio**: I pesi degli strati nascosti (o del bacino/reservoir) vengono **inizializzati casualmente e CONGELATI**.
* **Addestramento**: Si addestrano solo i pesi del layer di output tramite una semplice regressione lineare chiusa (LMS / Ridge Regression). Vantaggio: velocità di addestramento istantanea e assenza di minima locali.

---

### 4. Self-Organizing Maps (SOM di Kohonen)
* **Apprendimento Non Supervisionato Topologico**: Mappa dati ad alta dimensione su una griglia a 2D.
* **Regola di Aggiornamento del Neurone Vincitore (BMU) e dei Vicini**:
  $$w_i(t+1) = w_i(t) + \eta(t) h_{ic}(t) \big( x(t) - w_i(t) \big)$$
  dove $h_{ic}(t)$ è la funzione di vicinato gaussiana che decresce con la distanza dal BMU.

---

### 5. Graph Neural Networks (Message Passing)
* **Messaggio dal Vicinato**:
  $$m_v^{(k)} = \text{AGGREGATE}^{(k)} \left( \left\{ h_u^{(k-1)} : u \in N(v) \right\} \right)$$
* **Aggiornamento dello Stato del Nodo**:
  $$h_v^{(k)} = \text{UPDATE}^{(k)} \left( h_v^{(k-1)}, m_v^{(k)} \right)$$

---

### 6. Fenomeni Moderni: Double Descent & Lottery Ticket Hypothesis
* **Double Descent**: All'aumentare della complessità del modello, l'errore di test prima decresce, poi sale vicini al limite di interpolazione (overfitting classico), e poi **scende di nuovo** quando il modello diventa fortemente sopra-parametrizzato (over-parametrized regime).
* **Lottery Ticket Hypothesis (Frankle & Carbin)**: All'interno di una rete neurale grande inizializzata casualmente, esiste una sottorete ("biglietto vincente") che, se addestrata da sola con i pesi iniziali originari, raggiunge prestazioni pari all'intera rete in tempi minori.

---

# 📌 SEZIONE 6: Guida Integrale al Codice del Progetto (File per File)

Di seguito trovi la spiegazione dettagliata di come la teoria del corso è stata trasformata nel codice sorgente del repository del vostro gruppo.

---

### 6.1 I File Python di Supporto (`.py`)

#### 1. `cross_common.py` — *Motore di Validazione, Metriche e Nested CV*
* **Metriche**: Definisce le stringhe delle metriche (`MEE`, `MSE`, `RMSE`, `MAE`, `R2`) e la funzione `cr.mee` per calcolare il Mean Euclidean Error 4D per la CUP.
* **Prevenzione del Data Leakage (`SklearnRegressorRunner` e `SklearnNestedRegressorRunner`)**:
  * Incapsula i modelli di Scikit-Learn. Ad ogni fold del K-Fold, clona la pipeline ed esegue il `fit` **solo sul training fold del corrente K-Fold**, ed il `transform` sul validation fold. In questo modo lo scaling (`RobustScaler` / `StandardScaler`) non "vede" mai i dati di validazione in anticipo.
* **`kfold()` per Scikit-Learn**: Cicla sui fold, esegue l'addestramento ed estrae le metriche sia di Train che di Validation.
* **Nested Cross-Validation (`assess_sklearn_cv_robustness`)**:
  * Implementa il doppio ciclo (Inner Loop per la ricerca iperparametri con GridSearch, Outer Loop per la stima dell'errore di generalizzazione non polarizzato su fold esterni non usati nella scelta del modello).
* **Bootstrap (`generate_bootstrap_samples`)**: Genera campioni con reinserimento per stimare empiricamente l'intervallo di confidenza delle metriche.

#### 2. `nn_common.py` — *Architettura Rete Neurale PyTorch e Training Loop Custom*
* **`MLP(nn.Module)`**: Classe della rete neurale PyTorch. Costruisce una sequenza di layer lineari (`nn.Linear`) con attivazioni (`ReLU`, `Tanh`, `GELU`) ed accoppia un `OutputAdapter`.
* **`OutputAdapter`**:
  * `binary_bce_adapter`: Per MONK (classificazione binaria). Applica la Sigmoide ed esegue il thresholding $\ge 0.5$.
  * `regression_adapter`: Per la CUP (regressione multivariata 4D). Applica l'attivazione Identità $f(x) = x$.
* **`MEELoss(nn.Module)`**: Loss PyTorch personalizzata che calcola direttamente la distanza euclidea vettoriale $\frac{1}{N} \sum \|\hat{y} - y\|_2$ per permettere la retropropagazione diretta del MEE durante il backpropagation.
* **`train(...)` — Il Loop di Addestramento PyTorch Custom**:
  * Cicla per `epochs` epoche. Ad ogni batch:
    1. `optimizer.zero_grad()` per azzerare i gradienti accumulati.
    2. Forward pass: `output = model.forward(X)`.
    3. Calcolo della loss ed esecuzione del `loss.backward()`.
    4. Tracciamento della norma del gradiente (`gradient_norm(model)`).
    5. Aggiornamento pesi con `optimizer.step()`.
  * **Early Stopping (`EarlyStoppingStrategy`)**: Controlla la loss di validazione ad ogni epoca. Se per `patience` epoche la loss non migliora di almeno `min_delta`, interrompe l'addestramento e **ripristina i pesi salvati nello stato migliore (`best_state`)**.
  * **Learning Rate Decay**: Gestito tramite `ReduceLROnPlateau` che dimezza il tassi di apprendimento quando la loss sul validation stagna.
* **`kfold()` per PyTorch**: Applica la K-Fold Cross Validation alle reti PyTorch usando `ManualSplitStrategy`.

#### 3. `monk_common.py` e `cup_common.py` — *Data Loaders*
* **`monk_common.py`**: Carica i file dei 3 problemi MONK. Converte i 6 attributi simbolici $a_1 \dots a_6$ in **17 colonne binarie (0/1)** usando `pd.get_dummies` (1-of-k / One-Hot Encoding).
* **`cup_common.py`**: Carica `ML-CUP25-TR.csv` (500 campioni, 12 feature, 4 target) e `ML-CUP25-TS.csv` (1000 campioni del blind test). Gestisce la separazione feature/target e l'applicazione corretta dello scaling.

#### 4. `svm_common.py` e `knn_common.py` — *Helper Analitici*
* **`svm_common.py`**: Contiene `compare_svfreq_vs_permutation(...)` che confronta quali feature diventano più frequentemente Vettori di Supporto ($\alpha_i > 0$) rispetto alla *Permutation Importance* calcolata permutando le colonne.
* **`knn_common.py`**: Contiene `plot_knn_learning_curves_grid(...)` per plottare in una griglia di subplot l'andamento dell'errore al variare del numero dei vicini $k$.

---

### 6.2 I Notebook Jupyter del Progetto (`notebooks/`)

Tutti i notebook seguono un **flusso di lavoro rigoroso in 7 passaggi**:

```mermaid
flowchart TD
    S1["1. Introspezione & Seed"] --> S2["2. Setup GridSearch / Optuna"]
    S2 --> S3["3. Model Selection (Trova Iperparametri)"]
    S3 --> S4["4. Model Assessment (CV & Nested CV)"]
    S4 --> S5["5. Analisi Robustezza & Bootstrap"]
    S5 --> S6["6. Addestramento Modello Finale & Learning Curves"]
    S6 --> S7["7. Predizione Blind Test Set (CUP)"]
```

#### A) I Notebook per MONK (Classificazione)
1. **`monk_KNN.ipynb`**: Valuta KNN sui 3 problemi MONK. Risolve MONK-1 al 100% con $k=1$, ma mostra che **KNN fatica su MONK-2** (~75% accuracy) perché la regola di conteggio globale non è cogliibile dalle metriche di distanza locali.
2. **`monk_SVM.ipynb`**: Applica le SVM. Con kernel polinomiale di grado 2 o RBF risolve **al 100% sia MONK-1 che MONK-2**. L'analisi della frequenza dei Support Vector dimostra che gli attributi $a_1, a_2, a_5$ sono quelli guida.
3. **`monk_NN_optuna.ipynb`**: Esplora lo spazio degli iperparametri della Rete Neurale PyTorch con Optuna (cercando architettura, learning rate, weight decay).
4. **`monk_NN_1.ipynb`**: **Il Notebook di Collaudo Richiesto da Micheli**. Addestra una piccola rete PyTorch (2-4 neuroni nascosti) su MONK-1 raggiungendo il **100.0% di Accuracy sia in Train che in Test** con convergenza fluida.
5. **`monk_NN_2.ipynb`** e **`monk_NN_3.ipynb`**: Completano l'addestramento ed i plot delle reti neurali sui problemi MONK-2 e MONK-3 (gestendo il rumore tramite regolarizzazione).

#### B) I Notebook per la ML CUP (Regressione Multivariata 4D)
1. **`cup_data_introspection.ipynb`**: Analizza il dataset della CUP (500 righe, 12 input, 4 target). Calcola il modello di riferimento banale (**`DummyRegressor`**), che predice la media dei target ed ottiene un **MEE Baseline di 35.79**.
2. **`cup_KNN.ipynb`**: KNN Regressor con `RobustScaler` e $k=3$. Ottiene un **MEE in CV di 16.07** (miglior modello classico).
3. **`cup_LinearSVR.ipynb`**: SVR Lineare. Ottiene un MEE di **25.50**, dimostrando l'assoluta necessità di passare a kernel non lineari.
4. **`cup_SVM.ipynb`**: SVR con Kernel RBF. Ottimizzando $C$, $\gamma$ ed $\epsilon$, ottiene un MEE di **20.98**.
5. **`cup_NN_optuna.ipynb`**: Esegue la ricerca Bayesiana con Optuna per la Rete Neurale sulla CUP. Trova che un'architettura a 2 layer nascosti `[256, 128]`, attivazione `tanh`, ottimizzatore `AdamW` e regolarizzazione $L_2$ raggiunge le prestazioni migliori.
6. **`cup_NN_manual.ipynb`**: Addestra la Rete Neurale finale PyTorch sulla CUP per tracciare le **Learning Curves** (loss ed MEE in funzione delle epoche) ed escludere overfitting.
7. **`cup_Ensemble.ipynb`**: Esperimento avanzato di **Ensemble Modeling** che combina le predizioni di più modelli (Rete Neurale + SVR + KNN) per ridurre la varianza e produrre il file finale `Celati_Degliotti_Melaccio_ML-CUP25-TS.csv`.
