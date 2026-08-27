# Mappa Concettuale e Compendio delle Formule: Machine Learning
**Corso del Prof. Alessio Micheli — Università di Pisa**
*(Guida Integrale Mappata 1:1 sulle Slide Ufficiali del Corso [ML-25], Teoria, Formule, Codice di Progetto e Dimostrazioni)*

---

## 🗺️ Mappa Concettuale Globale (Strutturata sulle Slide Ufficiali di Micheli)

```mermaid
flowchart TD
    ML["MACHINE LEARNING [ML-25]"] --> P1["1. INTRODUZIONE, MODELLI LINEARI & k-NN\n(Slides: INTRO, linear, knn)"]
    ML --> P2["2. RETI NEURALI MULTI-LAYER & BACKPROP\n(Slides: NN-part1, Backpropagation notes, NN-part2)"]
    ML --> P3["3. VALIDAZIONE & MODEL SELECTION\n(Slides: Valid1, Valid2, Valid3)"]
    ML --> P4["4. STATISTICAL LEARNING THEORY, SVM & BIAS-VARIANZA\n(Slides: SLT1, SVM, SVM-other info, Bias-Variance)"]
    ML --> P5["5. ARCHITTETURE AVANZATE, UN-SUPERVISED & DEEP LEARNING\n(Slides: CNN, Deep, Rand, Unsupervised-SOM, RNN, SDL-Intro)"]
    ML --> P6["6. GUIDA INTEGRALE AL CODICE DEL PROGETTO\n(File .py e Notebook MONK / CUP)"]
    ML --> P7["7. DOMANDE TRABOCCHETTO DELL'ORALE\n(Domande 'Spicy' di Micheli)"]
    ML --> P8["8. COMPENDIO ESAME SCRITTO & ORALE\n(8 Dimostrazioni Matematiche Complete)"]

    P1 --> F1["1.1 Framework 4 Elementi & Rischi R(h) / Remp(h)"]
    P1 --> F2["1.2 Perceptron, Novikoff, LMS, Equazioni Normali, Cover"]
    P1 --> F3["1.3 k-Nearest Neighbors (k-NN) & Curse of Dimensionality"]

    P2 --> N1["2.1 Activations (Sigmoid, Tanh, ReLU, ELU, Non-linearità)"]
    P2 --> N2["2.2 Universal Approximation & Architettura MLP"]
    P2 --> N3["2.3 Algoritmo di Backpropagation & Complessità O(|W|)"]
    P2 --> N4["2.4 Iperparametri (eta, alpha, lambda) & Optimizers"]

    P3 --> V1["3.1 Schemi CV (Hold-Out, K-Fold)"]
    P3 --> V2["3.2 Prevenzione Data Leakage (Scaler in-fold)"]
    P3 --> V3["3.3 Model Selection vs Assessment & Nested CV"]

    P4 --> S1["4.1 Statistical Learning Theory (VC-Dim, Radon, SRM)"]
    P4 --> S2["4.2 Support Vector Machines (Max/Soft Margin, KKT, Kernel RBF)"]
    P4 --> S3["4.3 Support Vector Regression (SVR, epsilon-tube)"]
    P4 --> S4["4.4 Decomposizione Bias-Varianza-Rumore"]

    P5 --> A1["5.1 Convolutional Neural Networks (CNN & im2col)"]
    P5 --> A2["5.2 Deep Learning (Vanishing Gradient & No-Flattening)"]
    P5 --> A3["5.3 Algoritmi Costruttivi (Cascade Cor) & Random NN (ESN)"]
    P5 --> A4["5.4 Unsupervised (K-Means, Voronoi, SOM) & Ensemble"]
    P5 --> A5["5.5 Recurrent NN (RNN) & Graph NN (GNN / Message Passing)"]

    P6 --> C1["File Python (.py) & Notebook MONK/CUP"]
    P7 --> T1["8 Domande Trabocchetto da Lode con Risposta Modello"]
    P8 --> D1["8 Dimostrazioni Passo-Passo (Backprop, Bias-Var, Novikoff, SVM, Radon)"]
```

---

# 📌 SEZIONE 1: Introduzione, Modelli Lineari & k-NN (Slides `INTRO`, `linear`, `knn`)

### 1.1 Il Framework di Machine Learning & Rischi (Slide `INTRO-v0.2`)
Ogni algoritmo di Machine Learning viene analizzato attraverso **4 elementi fondamentali**:

1. **I Dati ($D$) ed i Rischi (Generale vs Empirico)**:
   * Insieme delle osservazioni $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$, con $x_i \in \mathbb{R}^D$ e $y_i \in \mathcal{Y}$.
   * Suddivisione disgiunta: **Training Set** (addestramento pesi), **Validation Set** (Model Selection) e **Test Set** (Model Assessment).
   * **Rischio Generale / Errore Atteso ($R(h)$)**: L'errore medio teorico del modello sull'intera distribuzione congiunta $P(x, y)$ di tutti i possibili dati futuri (non direttamente calcolabile):
     $$R(h) = \int_{\mathcal{X} \times \mathcal{Y}} L(h(x), y) \, dP(x, y)$$
   * **Rischio Empirico ($R_{emp}(h)$)**: L'errore medio calcolato concretamente sul dataset osservato di $N$ campioni:
     $$R_{emp}(h) = \frac{1}{N} \sum_{i=1}^N L(h(x_i), y_i)$$

2. **Lo Spazio delle Ipotesi ($\mathcal{H}$) & Inductive Bias**:
   * L'insieme di tutte le funzioni $h: \mathcal{X} \to \mathcal{Y}$ rappresentabili dal modello scelto.
   * **Inductive Bias (Bias Induttivo)**: Assunzioni a priori per generalizzare su dati non visti. Si divide in:
     * **Language / Restriction Bias**: Limitazione dello spazio delle ipotesi $\mathcal{H}$ (es. assumere un iperpiano separatore).
     * **Search / Preference Bias**: Preferenza accordata dall'ottimizzatore a certe ipotesi in $\mathcal{H}$ (es. favorire pesi piccoli).
   * **Learner Privo di Bias (Unbiased Learner)**: Può classificare solo i punti già visti nel training set (comportandosi come una **Lookup Table**), ed è totalmente incapace di generalizzare.

3. **La Loss Function ($\mathcal{L}$)**:
   * Misura la discrepanza tra l'output predetto $h(x)$ e l'etichetta reale $y$.
   * **MSE Loss (Mean Squared Error)**: $L_{MSE}(w) = \frac{1}{N} \sum_{i=1}^N (y_i - h(x_i))^2$
   * **MEE Loss (Mean Euclidean Error — Target Multivariato ML CUP)**: $L_{MEE}(w) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2 = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^4 (y_{i,m} - \hat{y}_{i,m})^2}$
   * **BCE Loss (Binary Cross-Entropy)**: $L_{BCE}(w) = -\frac{1}{N} \sum_{i=1}^N \big[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \big]$

4. **L'Algoritmo di Ottimizzazione**:
   * Procedura numerica per minimizzare la loss: **Gradient Descent** $w^{(t+1)} = w^{(t)} - \eta \nabla L(w^{(t)})$.

---

### 1.2 Modelli Lineari: Perceptron, LMS & Equazioni Normali (Slide `linear-v0.1`)

#### A) Il Perceptron (Rosenblatt 1958)
* **Modello**: Unità a Soglia Lineare (LTU) con attivazione a gradino $\text{sign}(z)$:
  $$f(x) = \text{sign}(w^T x + b) = \begin{cases} +1 & \text{se } w^T x + b \ge 0 \\ -1 & \text{se } w^T x + b < 0 \end{cases}$$
* **Iperpiano di Decisione**: La superficie $w^T x + b = 0$ divide lo spazio in due semispazi. Il vettore $w$ è ortogonale all'iperpiano e punta verso il semispazio positivo (+1).
* **Regola di Aggiornamento On-Line**: Ad ogni errore ($o_i \neq y_i$): $w \leftarrow w + \eta (y_i - o_i) x_i$.
* **Teorema di Convergenza (Novikoff 1962)**: Se un dataset è linearmente separabile con margine $\gamma > 0$, l'algoritmo convergerà in al massimo $k \le \frac{R^2}{\gamma^2}$ passi ($R = \max_i \|x_i\|$).
* **Limiti del Perceptron**: Non può risolvere problemi non separabili come l'**XOR** (Minsky & Papert 1969).

#### 💡 Esempio d'Esame: Porta NOT con 1 Perceptrone
* Input $x \in \{0, 1\}$, Target $y = 1 - x$ (in etichette $\{-1, +1\}$: $x=0 \to y=+1$; $x=1 \to y=-1$).
* Pesi $w = -2$, bias $b = +1 \implies f(x) = \text{sign}(-2x + 1)$ (Corretto per sia $x=0$ che $x=1$).

---

#### B) LMS (Least Mean Squares) / Regola del Delta (Widrow-Hoff)
* Calcola l'errore sulla risposta continua $net_i = w^T x_i + b$ prima della soglia: $\Delta w = \eta (y_i - net_i) x_i$.
* Esegue la discesa del gradiente sulla superficie parabolica dell'MSE, ammettendo **un unico minimo globale**.

---

#### C) Soluzione in Forma Chiusa della Regressione Lineare (Equazioni Normali)
Per la regressione lineare MSE, il vettore pesi ottimo $w^*$ si ricava analiticamente risolvendo le **Equazioni Normali**:
$$w^* = (X^T X)^{-1} X^T y = X^+ y$$
dove $X^+ = (X^T X)^{-1} X^T$ è la **Matrice Pseudoinversa di Moore-Penrose**.

---

#### D) Linear Basis Expansion (LBE) & Teorema di Cover (1965)
* **Linear Basis Expansion (LBE)**: Proietta gli input $x \in \mathbb{R}^D$ in uno spazio di feature $\phi(x) \in \mathbb{R}^K$ ($K > D$):
  $$h(x) = w^T \phi(x) + b = \sum_{k=1}^K w_k \phi_k(x) + b$$
* **Teorema di Cover (1965)**: Un problema di classificazione non separabile in uno spazio a bassa dimensione ha un'elevata probabilità di diventare **linearmente separabile** se proiettato in modo non lineare in uno spazio a dimensione elevata ($K \gg D$).

---

### 1.3 k-Nearest Neighbors (k-NN) & Curse of Dimensionality (Slide `knn-v0.1`)
* **Instance-Based / Lazy Learning**: Non addestra parametri liberi durante la fase di train, ma memorizza l'intero dataset.
* **Formulazione Formale**: Sia $N_K(x)$ l'insieme dei $K$ vicini più prossimi secondo una metrica (es. Euclidea):
  * **Regressione k-NN (Media dei Vicini)**:
    $$\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$
  * **Classificazione k-NN (Voto a Maggioranza)**:
    $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$$
* **Maledizione della Dimensionalità (*Curse of Dimensionality*)**: Ad alta dimensione la densità dei dati crolla, i vicinati perdono la loro natura "locale" e la nozione di similarità sfuma.

---

# 📌 SEZIONE 2: Reti Neurali Multi-Layer & Backprop (Slides `NN-part1`, `NN-part2`)

### 2.1 Funzioni di Attivazione e loro Proprietà
Le attivazioni introducono la **non-linearità** essenziale nelle reti neurali:

1. **Sigmoide (Logistica)**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, derivata $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Satura per $|z| > 4$ (Vanishing Gradient).
2. **Tanh (Tangente Iperbolica)**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, derivata $\tanh'(z) = 1 - \tanh^2(z)$. È **zero-centered**, velocizzando la convergenza.
3. **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$, derivata $f'(x) = \mathbb{I}(x > 0)$. Agisce come un **interruttore / maschera binaria (0/1)**. Risolve il Vanishing Gradient per $x > 0$. Soluzione al Dying ReLU: **LeakyReLU** $f(x) = \max(\alpha x, x)$.
4. **ELU (Exponential Linear Unit)**: $f(x) = x$ per $x > 0$ e $\alpha (e^x - 1)$ per $x \le 0$. Mantiene la media delle attivazioni vicina allo zero.

#### ❓ Perché NON possiamo usare solo attivazioni lineari?
Con attivazioni lineari $f(z) = c \cdot z$, l'uscita complessiva crollerebbe in un **singolo modello lineare**: $o = W_L \dots W_1 x = W_{tot} x$.

---

### 2.2 Architettura MLP & Teorema di Approssimazione Universale
* **Teorema di Approssimazione Universale (Hornik/Cybenko 1989)**: Una rete feed-forward con 1 strato nascosto ed attivazioni continue non lineari può approssimare qualsiasi funzione continua su un insieme compatto con un grado di precisione arbitrario $\epsilon > 0$.

---

### 2.3 L'Algoritmo di Backpropagation (Slide `Backpropagation notes`)
1. **Forward Pass**: $net_j = \sum_u w_{ju} o_u$, $o_j = f_j(net_j)$.
2. **Backward Pass (Chain Rule)**:
   * **Neurone di Output $k$**: $\delta_k = (d_k - o_k) f'_k(net_k)$
   * **Neurone Nascosto $j$**: $\delta_j = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$
3. **Aggiornamento Pesi**: $\Delta w_{tu} = \eta \delta_t o_u$. Efficienza **$O(|W|)$** (lineare nel numero dei pesi).

---

### 2.4 Iperparametri di Addestramento & Inizializzazione (Slide `NN-part2-v0.1`)
$$\Delta w_{tu}(t) = -\eta \frac{\partial E}{\partial w_{tu}(t)} + \alpha \Delta w_{tu}(t-1) - \eta \lambda w_{tu}(t)$$
1. **Learning Rate ($\eta$)**: Tasso di apprendimento.
2. **Momentum ($\alpha \in [0, 1)$)**: Inerzia fisica contro le oscillazioni (Nesterov NAG valuta il gradiente nel punto futuro $w + \alpha \Delta w_{old}$).
3. **Weight Decay / Regolarizzazione $L_2$ ($\lambda$)**: Penalizza la norma al quadrato dei pesi $\frac{\lambda}{2} \|w\|^2$.
4. **Inizializzazione Pesi**: Inizializzare a zero distrugge la simmetria dei gradienti.
5. **Ottimizzatori Avanzati**: R-Prop (solo segno del gradiente), Mini-Batch, SGD, Adam, AdamW.

---

# 📌 SEZIONE 3: Validazione & Model Selection (Slides `Valid1`, `Valid2`, `Valid3`)

### 3.1 Schemi di Validazione Sperimentale (Slide `Valid1-v.0.1`)
* **Hold-Out**: Divisione semplice Train/Validation/Test.
* **K-Fold Cross-Validation (Slide `Valid2`)**: Suddivide il dataset in $K$ fold ruotando il blocco di validazione. Restituisce media e deviazione standard:
  $$\bar{E}_{CV} = \frac{1}{K} \sum_{k=1}^K E_k, \quad \sigma_{CV} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (E_k - \bar{E}_{CV})^2}$$

---

### 3.2 Prevenzione del Data Leakage (Slide `Valid2-v0.1`)
* **Regola Tassativa**: Qualunque trasformazione sui dati (es. `RobustScaler`, `StandardScaler`) **deve eseguire il `fit_transform` ESCLUSIVAMENTE sul training fold di ciascun ciclo di K-Fold**.

---

### 3.3 Model Selection vs Model Assessment & Nested CV (Slide `Valid3-v.0.1`)
* **Model Selection**: Ricerca della migliore configurazione di iperparametri $h^*$ effettuata sul Validation Set.
* **Model Assessment**: Valutazione finale dell'errore di generalizzazione su un Test Set indipendente.
* **Nested Cross-Validation (Doppio Ciclo)**:
  * **Inner Loop (K-Fold Interno)**: Dedicato alla **Model Selection** (GridSearch / Optuna su $K-1$ fold).
  * **Outer Loop (K-Fold Esterno)**: Dedicato al **Model Assessment** (valutazione non polarizzata / *unbiased* su fold mai visti in GridSearch).

---

# 📌 SEZIONE 4: Statistical Learning Theory, SVM & Bias-Varianza (Slides `SLT1`, `SVM`, `Bias-Variance`)

### 4.1 Statistical Learning Theory (SLT) & VC-Dimension (Slide `SLT1-v.0.1`)
* **Shattering (Frammentazione)**: Un insieme di $N$ punti è shatterato da $\mathcal{H}$ se $\mathcal{H}$ può realizzare tutte le $2^N$ dicotomie binaria (+1 / -1).
* **VC-Dimension ($h_{VC}$)**: La MASSIMA cardinalità $N$ per cui esiste ALMENO UN insieme di $N$ punti shatterabile.
* **VC-Dimension degli Iperpiani Lineari ($VC = D + 1$)**: Dimostrato tramite il **Teorema di Radon** ($\text{Conv}(A) \cap \text{Conv}(B) \neq \emptyset$), che prova l'impossibilità di shatterare $D+2$ punti in $\mathbb{R}^D$.
* **Structural Risk Minimization (SRM) & Bound di Vapnik**:
  $$R(h) \le R_{emp}(h) + \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$
  SRM seleziona l'ipotesi che minimizza il bound complessivo lungo gerarchie annidate $\mathcal{H}_1 \subset \mathcal{H}_2 \dots$. L'**Early Stopping** agisce come regolarizzazione SRM limitando la norma pesi e la dimensione VC effettiva.

---

### 4.2 Support Vector Machines (SVM & SVR) (Slide `SVM-v0.1`)
* **Massimo Margine Geometrico**: $M = \frac{2}{\|w\|}$. Massimizzare $M$ equivale a minimizzare $\frac{1}{2} \|w\|^2$.
* **Soft Margin SVM**: Introduce le slack variables $\xi_i \ge 0$: $\min \frac{1}{2} \|w\|^2 + C \sum \xi_i$ sotto vincoli $y_i(w^T x_i + b) \ge 1 - \xi_i$. ($C \to \infty$ Hard Margin; $C$ piccolo tollera più violazioni ed amplia il margine).
* **Formulazione Duale & KKT**: $\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$. Sparsità KKT: solo per i Vettori di Supporto si ha $\alpha_i > 0$.
* **Kernel Trick & RBF Gaussiano**: Proietta i dati in uno spazio a dimensione infinita tramite $K(x, z) = \exp(-\gamma \|x-z\|^2)$.
* **Support Vector Regression (SVR)**: Loss $\epsilon$-insensitive $L_\epsilon(y, f(x)) = \max(0, |y - f(x)| - \epsilon)$ con tubo di tolleranza $\pm \epsilon$ e slack doppie $(\xi_i, \xi_i^*)$.

---

### 4.3 Decomposizione Bias-Varianza-Rumore (Slide `Bias-Variance-v0.1`)
L'Errore Quadratico Medio atteso si scompone analiticamente in 3 componenti:
$$\mathbb{E}[(y - h_D(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D [ (h_D(x) - \bar{h}(x))^2 ]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$
* **Bias$^2$**: Errore di approssimazione del modello (**Underfitting** se alto).
* **Varianza**: Sensibilità del modello alle fluttuazioni dello specifico training set (**Overfitting** se alta).
* **Rumore Irriducibile ($\sigma^2$)**: Incertezza stocastica dei dati.

---

# 📌 SEZIONE 5: Architetture Avanzate, Un-Supervised & Deep Learning (Slides `CNN`, `Deep`, `Rand`, `Unsupervised-SOM`, `RNN`, `SDL-Intro`)

### 5.1 Convolutional Neural Networks (CNN) (Slide `NN-part3-CNN-v0.1`)
* **Operatori Convoluzionali**: Convoluzione 2D $(I * K)[i,j]$ vs Cross-Correlazione 2D in PyTorch:
  $$(I \star K)[i, j] = \sum_m \sum_n I[i+m, j+n] K[m, n]$$
  Parallelizzata su GPU tramite la trasformazione matriciale `im2col` (Image-to-Column).

---

### 5.2 Deep Learning & Risultati di Profondità (Slide `NN-part3-Deep-v0.1`)
* **Vanishing Gradient Chain Rule**: $\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$.
* **No-Flattening**: La funzione di parità a $N$-bit richiede $O(2^N)$ porte in 2 strati vs $O(N)$ in $\log N$ strati.
* **Double Descent & Lottery Ticket Hypothesis**: Fenomeno di seconda discesa dell'errore nel regime fortemente sovra-parametrizzato e sottoreti vincenti isolate.

---

### 5.3 Algoritmi Costruttivi & Random NN (Slide `NN-part3-Rand-v0.1`)
* **Cascade Correlation (Fahlman & Lebiere 1990)**: Aggiunge neuroni candidati ed li addestra per **massimizzare la covarianza $S$** con l'errore residuo ($+\eta \frac{\partial S}{\partial w}$). Congela i pesi d'ingresso per sempre (*frozen weights*) per evitare il *moving target problem*.
* **Random NN & Reservoir Computing (Echo State Networks ESN)**: Pesi del bacino casuali e congelati permanently; readout lineare addestrato in forma analitica chiusa.

---

### 5.4 Apprendimento Non Supervisionato: K-Means & SOM (Slide `Unsupervised-SOM-v.0.1`)
* **Quantizzazione Vettoriale & Voronoi**: Celle di Voronoi $V_i = \{x : \|x - \mu_i\| \le \|x - \mu_j\|\}$. Distorsione globale $E = \sum_i \sum_{x \in V_i} \|x - \mu_i\|^2$. Aggiornamento on-line $\mu_i \leftarrow \mu_i + \eta(x - \mu_i)$.
* **Self-Organizing Maps (SOM di Kohonen 1982)**: Griglia 2D topologica. Best Matching Unit (BMU) ed aggiornamento vicinato gaussiano $h_{ic}(t) = \exp\left(-\frac{\|r_i - r_c\|^2}{2\sigma(t)^2}\right)$.
* **Ensemble Learning (Voting, Bagging, Boosting)**: Comitato di modelli (media/maggioranza), Bagging per alta varianza (averaging), Boosting (AdaBoost, ripesaggio errori) per alto bias.

---

### 5.5 Recurrent Neural Networks (RNN) (Slide `RNN-v0.1`)
* **Transizione di Stato**: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
* **Gradient Clipping**: $g \leftarrow v \frac{g}{\|g\|_2}$ per evitare l'esplosione dei gradienti.

---

### 5.6 Domini Strutturati & Graph Neural Networks (Slide `SDL-Intro-v0.1`)
* **Message Passing per GNN**: $m_v^{(k)} = \text{AGGREGATE}\left(\{h_u^{(k-1)} : u \in N(v)\}\right)$, $h_v^{(k)} = \text{UPDATE}\left(h_v^{(k-1)}, m_v^{(k)}\right)$.
* **NN4G (Neural Network for Graphs)**: $h_v^{(k)} = f\left( W_1 x_v + \sum W_{2,l} \sum_{u \in N(v)} h_u^{(l)} \right)$.
* **Readout Globale**: $y_G = W_{out} \left( \sum_{v \in V} h_v \right)$.
* **Word Embeddings**: Relazioni geometrico-semantiche distribuiti ($\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$).

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

---

# 📌 SEZIONE 7: Domande Trabocchetto dell'Orale di Micheli (con Risposte Modello)

Di seguito trovi le **8 domande "spicy" più frequenti all'orale** con la risposta ideale da dare a voce per dimostrare il massimo livello di maturità scientifica.

---

### ❓ Q1: *"Perché nella regressione la Softmax non ha senso e si usa l'attivazione Identità?"*
* **Risposta Modello**: *"La funzione Softmax converte un vettore di logits in una distribuzione di probabilità i cui elementi sommano a 1, il che è perfetto per la classificazione multi-classe. Nella regressione (come per i 4 target della CUP), vogliamo stimare valori reali continui non vincolati ad una somma unitaria o ad un intervallo limitato $[0, 1]$, per cui l'output layer deve usare l'attivazione Identità $f(x) = x$."*

---

### ❓ Q2: *"Cos'è la Matrice di Gram nelle SVM e perché deve essere Semidefinita Positiva?"*
* **Risposta Modello**: *"La Matrice di Gram $G_{ij} = K(x_i, x_j)$ raccoglie i prodotti scalari trasformativi tra tutti i punti del dataset. Dal Teorema di Mercer, la condizione che $G$ sia semidefinita positiva ($v^T G v \ge 0 \,\, \forall v$) garantisce che la funzione obiettivo della formulazione duale sia strettamente convessa. Ciò assicura che il problema di programmazione quadratica ammetta UN SOLO MINIMO GLOBALE senza minimi locali."*

---

### 3️⃣ Q3: *"Perché la regola di aggiornamento pesi nella Cascade Correlation usa il segno PLUS ($+\eta \frac{\partial S}{\partial w}$)?"*
* **Risposta Modello**: *"Perché anziché minimizzare una funzione di perdita (dove si usa la discesa del gradiente col segno meno $-\eta \frac{\partial E}{\partial w}$), nella Cascade Correlation l'obiettivo del candidato è MASSIMIZZARE la covarianza $S$ con l'errore residuo della rete. Di conseguenza, l'aggiornamento dei pesi del candidato usa l'ascesa del gradiente col segno $+$. "*

---

### 4️⃣ Q4: *"Perché l'Early Stopping può essere formalmente considerato una forma di regolarizzazione di Tikhonov ($L_2$)?"*
* **Risposta Modello**: *"Arrestare l'addestramento ad un numero limitato di epoche $T$ impedisce ai pesi di crescere verso valori di norma elevati, limitando la loro magnitudo $\|w\|^2$. Questo agisce esattamente come il termine di penalità della regolarizzazione di Tikhonov $\frac{\lambda}{2} \|w\|^2$, riducendo la capacità teorica e la dimensione VC della rete."*

---

### 5️⃣ Q5: *"Perché la VC-Dimension di 1-NN è infinita, ma 1-NN può comunque generalizzare?"*
* **Risposta Modello**: *"1-NN memorizza perfettamente l'intero dataset creando tassellature di Voronoi attorno ad ogni punto. Ha $VC = \infty$ perché può separare qualsiasi combinazione di etichette $2^N$ su $N$ punti se distanziati. Tuttavia, la sua capacità di generalizzare su dati futuri non dipende da un vincolo sulla capacità dello spazio delle ipotesi, ma dalla liscia continuità della vera funzione target $f(x)$ sottostante (se punti vicini nello spazio delle feature appartengono alla stessa classe)."*

---

### 6️⃣ Q6: *"Perché l'inizializzazione dei pesi a zero distrugge l'addestramento di una Rete Neurale?"*
* **Risposta Modello**: *"Se tutti i pesi di un layer hidden vengono inizializzati a zero (o a qualsiasi valore costante uguale), tutti i neuroni dello strato calcoleranno esattamente lo stesso valore di attivazione $net$ nel forward pass e riceveranno esattamente lo stesso segnale d'errore $\delta$ durante il backpropagation. I pesi si aggiorneranno tutti della stessa quantità, mantenendo i neuroni identici tra loro ad ogni epoca e distruggendo la capacità della rete di apprendere feature differenti (rottura della simmetria dei gradienti)."*

---

### 7️⃣ Q7: *"Che cos'è la rappresentazione latente in un Autoencoder Undercomplete?"*
* **Risposta Modello**: *"È la codifica compressa a dimensionalità ridotta prodotta dallo strato intermedio bottleneck $z$ ($dim(z) < dim(x)$). Costringe la rete a scartare il rumore e la ridondanza dell'input, estraendo solo le feature latenti essenziali necessarie per ricostruire il dato originale nella fase di decoding."*

---

### 8️⃣ Q8: *"Perché lo scaling (es. StandardScaler o RobustScaler) va inserito dentro la Pipeline del K-Fold e non calcolato prima?"*
* **Risposta Modello**: *"Se calcolassimo la media e la deviazione standard sull'intero dataset prima della Cross-Validation, le informazioni del Validation set influenzerebbero la normalizzazione dei dati di addestramento. Questo genererebbe Data Leakage (contaminazione dei dati), portando a stime di errore ottimistiche e non veritiere. Inserendo lo scaler dentro la Pipeline, il `fit_transform` viene calcolato ESCLUSIVAMENTE sul training fold di ogni singolo ciclo."*

---

# 📌 SEZIONE 8: Compendio d'Esame — Domande Tipiche con Dimostrazioni Matematiche Complete

Di seguito sono riportate le **8 domande matematiche più importanti dello scritto e dell'orale**, svolte con tutti i passaggi algebrici rigorosi.

---

### 📐 D1: Dimostrazione Formale della Backpropagation (Chain Rule e Derivazione dei $\delta$)

**Domanda**: *"Derivare la regola di aggiornamento pesi $\Delta w_{tu}$ per un neurone generico usando la chain rule della propagazione all'indietro dell'errore. Distinguere il caso in cui $t$ sia un neurone di output rispetto ad un neurone nascosto."*

#### 1. Definizione dell'Errore e della Chain Rule
Per un singolo pattern $p$, definiamo l'errore quadratico medio:
$$E_p = \frac{1}{2} \sum_{k \in \text{Output}} (d_{p,k} - o_{p,k})^2$$

Il net input al neurone $t$ è dato dalla somma pesata delle uscite del layer precedente:
$$net_t = \sum_u w_{tu} o_u$$

L'uscita del neurone $t$ è $o_t = f_t(net_t)$. Applicando la Chain Rule per calcolare la derivata rispetto al peso $w_{tu}$:
$$\frac{\partial E_p}{\partial w_{tu}} = \frac{\partial E_p}{\partial net_t} \cdot \frac{\partial net_t}{\partial w_{tu}}$$

Poiché $\frac{\partial net_t}{\partial w_{tu}} = o_u$, definendo il **local error factor** $\delta_t = -\frac{\partial E_p}{\partial net_t}$, otteniamo:
$$\Delta w_{tu} = -\eta \frac{\partial E_p}{\partial w_{tu}} = \eta \delta_t o_u$$

#### 2. Caso A: Neurone $t$ nel Layer di Output ($t = k$)
L'errore $E_p$ dipende direttamente da $o_k$. Usiamo nuovamente la Chain Rule:
$$\delta_k = -\frac{\partial E_p}{\partial net_k} = -\frac{\partial E_p}{\partial o_k} \cdot \frac{\partial o_k}{\partial net_k}$$

* Derivata della loss rispetto all'output: $\frac{\partial E_p}{\partial o_k} = \frac{\partial}{\partial o_k} \frac{1}{2}(d_k - o_k)^2 = -(d_k - o_k)$.
* Derivata dell'output rispetto al net: $\frac{\partial o_k}{\partial net_k} = f'_k(net_k)$.

Sostituendo:
$$\delta_k = - \big( -(d_k - o_k) \big) f'_k(net_k) = (d_k - o_k) f'_k(net_k)$$

#### 3. Caso B: Neurone $t$ nel Layer Nascosto ($t = j$)
Un neurone nascosto $j$ influenza l'errore $E_p$ attraverso **tutti i neuroni $k$ del layer di output a cui è connesso**. Usiamo la derivata parziale multivariata:
$$\delta_j = -\frac{\partial E_p}{\partial net_j} = -\sum_{k \in \text{Output}} \frac{\partial E_p}{\partial net_k} \cdot \frac{\partial net_k}{\partial o_j} \cdot \frac{\partial o_j}{\partial net_j}$$

Riconoscendo che $-\frac{\partial E_p}{\partial net_k} = \delta_k$, che $\frac{\partial net_k}{\partial o_j} = \frac{\partial}{\partial o_j} \sum_j w_{kj} o_j = w_{kj}$, e che $\frac{\partial o_j}{\partial net_j} = f'_j(net_j)$:
$$\delta_j = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$$

---

### 📐 D2: Dimostrazione Algebrica della Scomposizione Bias-Varianza-Rumore

**Domanda**: *"Derivare analiticamente la decomposizione dell'Errore Quadratico Medio Atteso $\mathbb{E}[(y - h_D(x))^2]$ nelle componenti di Bias al quadrato, Varianza e Rumore Irriducibile."*

#### 1. Modello di Generazione dei Dati
Sia $y = f(x) + \epsilon$, dove $f(x)$ è la vera funzione target e $\epsilon$ è un rumore stocastico a media nulla e varianza $\sigma^2$:
$$\mathbb{E}[\epsilon] = 0, \quad \mathbb{E}[\epsilon^2] = \sigma^2, \quad \mathbb{E}[f(x)\epsilon] = 0$$

Sia $h_D(x)$ il modello appreso da un dataset $D$, e sia $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$ la media delle predizioni su tutti i possibili dataset.

#### 2. Riscrittura dello Scarto
Espandiamo il termine di errore $y - h_D(x)$ sommando e sottrando $\bar{h}(x)$:
$$y - h_D(x) = (f(x) + \epsilon) - h_D(x) = \underbrace{(f(x) - \bar{h}(x))}_{\text{A}} + \underbrace{(\bar{h}(x) - h_D(x))}_{\text{B}} + \underbrace{\epsilon}_{\text{C}}$$

#### 3. Calcolo del Valore Atteso del Quadrato $\mathbb{E}_{D, \epsilon} \big[ (A + B + C)^2 \big]$
$$(A + B + C)^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC$$

Calcoliamo il valore atteso termine per termine:
1. $\mathbb{E}_D [A^2] = \mathbb{E}_D [(f(x) - \bar{h}(x))^2] = (f(x) - \bar{h}(x))^2 = \text{Bias}(x)^2$ *(è una costante rispetto al dataset $D$)*.
2. $\mathbb{E}_D [B^2] = \mathbb{E}_D [(\bar{h}(x) - h_D(x))^2] = \text{Varianza}(x)$ *(definizione di varianza della stima)*.
3. $\mathbb{E}_\epsilon [C^2] = \mathbb{E}[\epsilon^2] = \sigma^2$ *(rumore irriducibile)*.
4. $2 \mathbb{E}_D [AB] = 2(f(x) - \bar{h}(x)) \mathbb{E}_D [\bar{h}(x) - h_D(x)] = 2(f(x) - \bar{h}(x)) (\bar{h}(x) - \bar{h}(x)) = 0$.
5. $2 \mathbb{E}_{D,\epsilon} [AC] = 2(f(x) - \bar{h}(x)) \mathbb{E}[\epsilon] = 0$.
6. $2 \mathbb{E}_{D,\epsilon} [BC] = 2 \mathbb{E}_D [\bar{h}(x) - h_D(x)] \mathbb{E}[\epsilon] = 0$.

Sommando i termini non nulli otteniamo la scomposizione esatta:
$$\mathbb{E}_{D, \epsilon} \big[ (y - h_D(x))^2 \big] = \text{Bias}(x)^2 + \text{Varianza}(x) + \sigma^2$$

---

### 📐 D3: Dimostrazione del Teorema di Convergenza del Perceptrone

**Domanda**: *"Enunciare e dimostrare il Teorema di Convergenza del Perceptrone per dataset linearmente separabili."*

#### 1. Ipotesi
Sia $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$ con $y_i \in \{-1, +1\}$ un dataset **linearmente separabile con margine $\gamma > 0$**.
Ciò significa che esiste un vettore pesi ideale $w^*$ unitario ($\|w^*\| = 1$) tale che:
$$y_i (w^{*T} x_i) \ge \gamma > 0 \quad \forall i=1 \dots N$$

Sia $R = \max_i \|x_i\|$ la massima norma degli esempi di input.
Sia $w_0 = 0$ il vettore pesi iniziale. La regola di aggiornamento all'errore $k$-esimo è $w_k = w_{k-1} + y_i x_i$.

#### 2. Dimostrazione del Limite Inferiore sul Prodotto Scalare $w_k^T w^*$
Consideriamo il prodotto scalare $w_k^T w^*$:
$$w_k^T w^* = (w_{k-1} + y_i x_i)^T w^* = w_{k-1}^T w^* + y_i (x_i^T w^*)$$

Poiché $y_i (w^{*T} x_i) \ge \gamma$, applicando la relazione ricorsivamente per $k$ aggiornamenti:
$$w_k^T w^* \ge w_{k-1}^T w^* + \gamma \ge w_{k-2}^T w^* + 2\gamma \dots \ge k \gamma$$
$$\implies w_k^T w^* \ge k \gamma \quad \text{(1)}$$

#### 3. Dimostrazione del Limite Superiore sulla Norma $\|w_k\|^2$
Consideriamo la norma al quadrato $\|w_k\|^2$:
$$\|w_k\|^2 = \|w_{k-1} + y_i x_i\|^2 = \|w_{k-1}\|^2 + 2 y_i (w_{k-1}^T x_i) + \|x_i\|^2$$

Poiché l'aggiornamento avviene **solo quando il Perceptrone commette un errore**, abbiamo $y_i (w_{k-1}^T x_i) \le 0$. Inoltre $\|x_i\|^2 \le R^2$:
$$\|w_k\|^2 \le \|w_{k-1}\|^2 + 0 + R^2 \le \|w_{k-2}\|^2 + 2 R^2 \dots \le k R^2$$
$$\implies \|w_k\| \le \sqrt{k} R \quad \text{(2)}$$

#### 4. Conclusione tramite Disuguaglianza di Cauchy-Schwarz
Per la disuguaglianza di Cauchy-Schwarz: $w_k^T w^* \le \|w_k\| \|w^*\| = \|w_k\|$ (dato che $\|w^*\|=1$).
Combinando le disuguaglianze (1) e (2):
$$k \gamma \le w_k^T w^* \le \|w_k\| \le \sqrt{k} R$$
$$k \gamma \le \sqrt{k} R \implies k^2 \gamma^2 \le k R^2 \implies k \le \frac{R^2}{\gamma^2}$$

Il numero massimo di errori $k$ prima di raggiungere la convergenza perfetta è **finito e limitato superiormente da $\frac{R^2}{\gamma^2}$**.

---

### 📐 D4: Support Vector Machines — Dal Margine Geometrico alla Formulazione Duale

**Domanda**: *"Ricavare il margine geometrico $M = \frac{2}{\|w\|}$, impostare il problema di ottimizzazione per le Soft Margin SVM e ricavare la formulazione duale tramite la Lagrangiana."*

#### 1. Derivazione del Margine Geometrico
Un iperpiano separatore è definito da $w^T x + b = 0$. La distanza ortogonale di un punto $x_i$ dall'iperpiano è $d_i = \frac{|w^T x_i + b|}{\|w\|}$.
Per garantire la corretta classificazione con margine, imponiamo la condizione di separazione canonica per i vettori di supporto:
$$y_i (w^T x_i + b) \ge 1$$
La distanza dei punti più vicini (Support Vectors) dall'iperpiano è $\frac{1}{\|w\|}$. La larghezza totale della fascia di margine tra le due classi è dunque:
$$M = \frac{2}{\|w\|}$$
Massimizzare il margine $M$ equivale a minimizzare $\|w\|$, ovvero minimizzare $\frac{1}{2} \|w\|^2$.

#### 2. Formulazione Primate Soft Margin (con Slack Variables $\xi_i$)
Per tollerare punti non linearmente separabili dentro il margine o errati, introduciamo le slack variables $\xi_i \ge 0$:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{sotto i vincoli } y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

#### 3. Derivazione della Formulazione Duale
Costruiamo la funzione Lagrangiana con moltiplicatori $\alpha_i \ge 0$ e $\mu_i \ge 0$:
$$\mathcal{L}(w, b, \xi, \alpha, \mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i \big[ y_i(w^T x_i + b) - 1 + \xi_i \big] - \sum_{i=1}^N \mu_i \xi_i$$

Azzeriamo le derivate parziali rispetto alle variabili primali $(w, b, \xi_i)$:
1. $\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^N \alpha_i y_i x_i$
2. $\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^N \alpha_i y_i = 0$
3. $\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies \alpha_i \le C$

Sostituendo queste tre relazioni nella Lagrangiana ed effettuando i passaggi algebrici, le componenti con $w$ e $\xi$ si semplificano, ottenendo la **Formulazione Duale**:
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i^T x_j) \quad \text{sotto } 0 \le \alpha_i \le C \text{ e } \sum_{i=1}^N \alpha_i y_i = 0$$

Sostituendo il prodotto scalare $x_i^T x_j$ con una funzione Kernel $K(x_i, x_j)$ si applica il **Kernel Trick**.

---

### 📐 D5: Cascade Correlation — Covarianza del Candidato e Congelamento dei Pesi

**Domanda**: *"Fornire l'espressione matematica della funzione obiettivo $S$ usata per addestrare un neurone candidato nella Cascade Correlation, derivare il suo gradiente e spiegare perché i pesi d'ingresso vengono congelati."*

#### 1. Funzione Obiettivo di Covarianza $S$
Nella Cascade Correlation, un neurone candidato $c$ riceve input da tutte le feature d'ingresso e da tutti i neuroni nascosti preesistenti. L'obiettivo è **massimizzare la covarianza $S$** tra l'uscita del candidato $o_p$ e l'errore residuo del layer di output $E_{p,k}$ su tutti i pattern $p$ ed i target $k$:
$$S = \sum_{k \in \text{Output}} \left| \sum_{p} (o_p - \bar{o})(E_{p,k} - \bar{E}_k) \right|$$
dove $\bar{o} = \frac{1}{N} \sum_p o_p$ è l'uscita media del candidato e $\bar{E}_k = \frac{1}{N} \sum_p E_{p,k}$ è l'errore medio sul target $k$.

#### 2. Ascesa del Gradiente
Per massimizzare $S$, eseguiamo l'ascesa del gradiente sui pesi d'ingresso $w_{j}$ del candidato:
$$\Delta w_{j} = +\eta \frac{\partial S}{\partial w_{j}}$$

Sia $\sigma_k = \text{sign}\left( \sum_{p} (o_p - \bar{o})(E_{p,k} - \bar{E}_k) \right)$ il segno della covarianza per il target $k$. Derivando rispetto a $w_j$:
$$\frac{\partial S}{\partial w_{j}} = \sum_{k} \sigma_k \sum_{p} (E_{p,k} - \bar{E}_k) f'_p(net_c) x_{p,j}$$

#### 3. Congelamento dei Pesi (Frozen Weights)
Una volta completato l'addestramento del candidato e massimizzata la covarianza $S$, il neurone viene inserito stabilmente nella rete ed **i suoi pesi d'ingresso vengono congelati per sempre**.
* **Motivazione teorica**: Evita il fenomeno del *moving target problem* (in cui i neuroni nascosti continuano a cambiare ruolo destabilizzando gli altri strati) e garantisce che ogni neurone agisca come un estrattore di feature permanente.

---

### 📐 D6: Support Vector Regression (SVR) ed $\epsilon$-Insensitive Loss

**Domanda**: *"Scrivere la funzione di perdita $\epsilon$-insensitive della SVR, impostare il problema di ottimizzazione con le variabili slack $(\xi_i, \xi_i^*)$ e chiarire le differenze rispetto alla loss MSE."*

#### 1. La $\epsilon$-Insensitive Loss
A differenza della regressione classica con MSE, la SVR utilizza la loss $\epsilon$-insensitive $|y - f(x)|_\epsilon$:
$$L_\epsilon(y, f(x)) = |y - f(x)|_\epsilon = \begin{cases} 0 & \text{se } |y - f(x)| \le \epsilon \\ |y - f(x)| - \epsilon & \text{altrimenti} \end{cases}$$
L'errore è pari a zero se il punto cade all'interno di un tubo ("tube") di ampiezza $\pm \epsilon$ attorno alla funzione di regressione predetta $f(x) = w^T x + b$.

#### 2. Formulazione Primate con Slack Variables $(\xi_i, \xi_i^*)$
Poiché le deviazioni possono avvenire sia al di sopra ($\xi_i$) che al di sotto ($\xi_i^*$) del tubo di tolleranza $\epsilon$:
$$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N (\xi_i + \xi_i^*)$$
$$\text{sotto i vincoli: } \begin{cases} y_i - (w^T x_i + b) \le \epsilon + \xi_i \\ (w^T x_i + b) - y_i \le \epsilon + \xi_i^* \\ \xi_i \ge 0, \,\, \xi_i^* \ge 0 \end{cases}$$

#### 3. Confronto SVR ($\epsilon$-insensitive) vs MSE
* **MSE ($L_2$)**: Penalizza tutti i residui in modo quadratico. Un singolo punto isolato lontano (outlier) ha un impatto enorme ed attrae a sé la curva di regressione.
* **SVR ($\epsilon$-insensitive)**: Ignora le piccole fluttuazioni minori di $\epsilon$ (garantendo la **sparsità dei Support Vector**) e penalizza i residui esterni in modo **lineare**, garantendo una forte robustezza agli outlier.

---

### 📐 D7: Nested Cross-Validation (Doppio Ciclo per Model Selection & Assessment)

**Domanda**: *"Spiegare l'architettura della Nested Cross-Validation, il ruolo dei due cicli Inner ed Outer loop e perché la CV semplice porta ad una stima polarizzata dell'errore."*

```mermaid
flowchart TD
    D["Dataset Complessivo (N campioni)"] --> OL["OUTER LOOP (5-Fold Outer CV)\n[Stima Errore di Generalizzazione (Model Assessment)]"]
    OL --> |Per ogni Outer Fold| IL["INNER LOOP (3-Fold Inner CV sul Train Esterno)\n[Scelta Iperparametri Migliori (Model Selection)]"]
    IL --> |GridSearch su Inner Folds| BestH["Seleziona Iperparametri h*"]
    BestH --> Retrain["Addestra Modello con h* sul solo Train Esterno"]
    Retrain --> TestExt["Valuta su Test Fold dell'Outer Loop (Mai visto in GridSearch!)"]
    TestExt --> FinalEst["Media Errore sui 5 Outer Folds = Stima Non Polarizzata"]
```

#### 1. Limite della Cross-Validation Semplice
Se utilizziamo la K-Fold Cross Validation sia per selezionare la combinazione di iperparametri migliore $h^*$ sia per riportare l'errore finale, l'errore riportato sul validation set per $h^*$ soffre di **overfitting da ricerca iper-parametri** (*selection bias*). La stima dell'errore risulterà ottimistica (polarizzata).

#### 2. Architettura della Nested Cross-Validation
Per eliminare la polarizzazione si separano nettamente i due processi con due cicli annidati:
1. **Outer Loop (es. 5-Fold)**: Divide il dataset originale in 5 blocchi. In ogni iterazione, 1 blocco viene congelato come **Test Set Esterno** e 4 blocchi costituiscono il **Training Set Esterno**.
2. **Inner Loop (es. 3-Fold)**: Prende il solo Training Set Esterno e lo suddivide ulteriormente per eseguire la GridSearch o l'ottimizzazione degli iperparametri, trovando la configurazione migliore $h^*$.
3. **Valutazione Non Polarizzata**: Si ri-addestra il modello con $h^*$ sull'intero Training Set Esterno e lo si valuta sul **Test Set Esterno** (che non ha partecipato alla scelta degli iperparametri).
4. La media dell'errore sui 5 Test Set Esterni costituisce la reale stima non polarizzata delle prestazioni di generalizzazione.

---

### 📐 D8: Dimostrazione Formale della VC-Dimension degli Iperpiani in $\mathbb{R}^D$ ($VC = D+1$)

**Domanda**: *"Dimostrare formalmente che la VC-dimension degli iperpiani lineari in $\mathbb{R}^D$ è pari a $D+1$ usando il Teorema di Radon."*

#### 1. Parte I: Shattering di $D+1$ Punti (Limite Inferiore $VC \ge D+1$)
Dobbiamo mostrare che esiste ALMENO UN insieme di $D+1$ punti in $\mathbb{R}^D$ che può essere shatterato (ovvero che ammette tutte le $2^{D+1}$ dicotomie).

Scegliamo i seguenti $D+1$ punti canonici:
$$x_0 = (0, 0, \dots, 0)^T$$
$$x_1 = (1, 0, \dots, 0)^T, \quad x_2 = (0, 1, \dots, 0)^T, \quad \dots \quad x_D = (0, 0, \dots, 1)^T$$

Per qualsiasi assegnamento arbitrario di etichette $y_0, y_1, \dots, y_D \in \{-1, +1\}$, possiamo definire i parametri dell'iperpiano $w = (w_1, \dots, w_D)^T$ e $b$ come segue:
* Scegliamo $b = y_0$.
* Scegliamo $w_i = y_i - y_0$ per $i = 1 \dots D$.

Verifichiamo le predizioni $\text{sign}(w^T x_i + b)$:
* Per $x_0$: $\text{sign}(w^T x_0 + b) = \text{sign}(0 + b) = \text{sign}(y_0) = y_0$.
* Per $x_i$ ($i \ge 1$): $\text{sign}(w^T x_i + b) = \text{sign}(w_i + b) = \text{sign}((y_i - y_0) + y_0) = \text{sign}(y_i) = y_i$.

Poiché l'iperpiano può realizzare qualunque combinazione di etichette per questi $D+1$ punti, l'insieme è shatterato $\implies VC(\mathcal{H}) \ge D+1$.

#### 2. Parte II: Impossibilità di Shatterare $D+2$ Punti (Limite Superiore $VC < D+2$)
Dobbiamo mostrare che NESSUN insieme di $D+2$ punti in $\mathbb{R}^D$ può essere shatterato.

**Teorema di Radon**: Ogni insieme di $D+2$ punti $S = \{x_1, \dots, x_{D+2}\}$ in $\mathbb{R}^D$ può essere partizionato in due sottoinsiemi disgiunti $A$ e $B$ ($A \cup B = S, A \cap B = \emptyset$) tali che i loro **inviluppi convessi si intersecano**:
$$\text{Conv}(A) \cap \text{Conv}(B) \neq \emptyset$$

Esiste quindi un punto $x^*$ espresso come combinazione convessa sia dei punti in $A$ che dei punti in $B$:
$$x^* = \sum_{x_i \in A} \lambda_i x_i = \sum_{x_j \in B} \mu_j x_j, \quad \text{con } \sum \lambda_i = 1, \sum \mu_j = 1, \,\, \lambda_i, \mu_j \ge 0$$

Se assegnamo etichetta $+1$ a tutti i punti in $A$ e $-1$ a tutti i punti in $B$, supponiamo per assurdo che esista un iperpiano $w^T x + b$ che li separa:
* Per $x_i \in A \implies w^T x_i + b > 0 \implies \sum \lambda_i (w^T x_i + b) > 0 \implies w^T x^* + b > 0$.
* Per $x_j \in B \implies w^T x_j + b < 0 \implies \sum \mu_j (w^T x_j + b) < 0 \implies w^T x^* + b < 0$.

Otteniamo la contraddizione $w^T x^* + b > 0$ e $w^T x^* + b < 0$. Nessun iperpiano lineare può realizzare questa dicotomia su $D+2$ punti.

Di conseguenza, $D+2$ punti non possono MAI essere shatterati $\implies VC(\mathcal{H}) < D+2$.

**Conclusione**: $VC(\mathcal{H}) = D + 1$.
