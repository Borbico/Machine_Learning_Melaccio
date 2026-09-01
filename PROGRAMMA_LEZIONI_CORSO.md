# 📚 Programma e Indice Strutturato delle Lezioni: Machine Learning
**Corso del Prof. Alessio Micheli — Università di Pisa**  
*(Elenco Integrale Ordinato e Strutturato di Tutti gli Argomenti e Formule del Corso — Mappatura 1:1 sui File delle Slide [ML-25])*

---


---

## 💡 Introduzione al Corso e Filo Conduttore (La Visione d'Insieme)

Il corso di Machine Learning del Prof. Alessio Micheli è strutturato attorno a **una domanda fondamentale**:
> *"Come possiamo insegnare a un computer ad apprendere una funzione incognita dai dati, garantendo che sappia generalizzare bene su dati futuri mai visti senza limitarsi a imparare a memoria il training set?"*

Da questa domanda nascono i **5 grandi blocchi del corso**:

1. **Il Primo Passo — Dalle Basi ai Modelli Lineari e k-NN**:  
   Si parte dal *Framework dei 4 elementi* (Dati $D$, Spazio delle Ipotesi $\mathcal{H}$, Loss Function $\mathcal{L}$ e Ottimizzatore). Si comprende perché non si può fare ML senza *Bias Induttivo*. Si studiano i primi modelli storici: il *Perceptrone* (unità a soglia, Novikoff convergence bound e limite dell'XOR), la *Delta Rule (LMS)* per la discesa continua del gradiente, le *Equazioni Normali* per la regressione lineare chiusa ($w^* = X^+ y$), la *Linear Basis Expansion (LBE)* e il *Teorema di Cover*, fino al *k-NN* (instance-based) e la *Maledizione della Dimensionalità*.

2. **Il Cuore delle Reti Neurali — MLP & Backpropagation**:  
   Si esplora il mondo delle Multi-Layer Perceptrons: perché le attivazioni non lineari (Sigmoide, Tanh, ReLU, ELU) sono *obbligatorie* (altrimenti la rete collassa in un unico modello lineare), il *Teorema di Approssimazione Universale* (Hornik/Cybenko), e la *Backpropagation* (calcolo efficiente dei $\delta$ via Chain Rule con complessità $O(|W|)$), affrontando i problemi pratici di *Vanishing Gradient* ed i parametri di controllo (*Learning Rate $\eta$*, *Momentum $\alpha$*, *Weight Decay $\lambda$*).

3. **La Metodologia Sperimentale — Validazione senza Imbrogliare**:  
   Si definisce la rigorosa metodologia di valutazione per evitare il *Data Leakage* (lo scaler va addestrato esclusivamente sui fold di train). Si analizzano gli schemi di validazione (*Hold-Out*, *K-Fold Cross-Validation*) e si formalizza la distinzione tra *Model Selection* (scelta degli iperparametri sul Validation Set) e *Model Assessment* (stima unbiased del rischio sul Test Set) tramite la *Nested Cross-Validation* (doppio ciclo interno/esterno).

4. **La Matematica Forte — SLT (Vapnik), Bias-Varianza e SVM**:  
   La teoria formale dell'apprendimento: la decomposizione algebrica *Bias-Varianza-Rumore* (Underfitting vs Overfitting), la *Statistical Learning Theory (SLT)* di Vapnik per misurare la capacità del modello tramite la *VC-Dimension* ($VC = D+1$ per iperpiani via Teorema di Radon), ed il Bound di Vapnik per l'ottimizzazione del principio *SRM (Structural Risk Minimization)*. Infine le *Support Vector Machines (SVM & SVR)*: massimizzazione del margine ($M = 2/\|w\|$), Slack Variables ($C$), formulazione Duale con condizioni KKT, *Kernel Trick* (RBF Gaussiano) ed SVR con tubo $\epsilon$-insensitive.

5. **La Frontiera — Architetture Avanzate, Deep Learning e Domini Strutturati**:  
   La panoramica sui modelli evoluti: *CNN* (convoluzioni 2D, cross-correlazione e `im2col`), *Deep Learning & No-Flattening* (vantaggio esponenziale della profondità a $\log N$ strati), *Algoritmi Costruttivi e Random NN* (*Cascade Correlation* con covarianza $S$ e pesi congelati, *Echo State Networks ESN*), *Unsupervised Learning & SOM* (K-Means, Celle di Voronoi, Self-Organizing Maps di Kohonen, Autoencoders, Ensemble Bagging/Boosting), e la modellazione temporale e su grafi (*RNN* con BPTT e Gradient Clipping; *GNN* con Message Passing, NN4G e Readout Globale).

## 🗺️ Struttura dei Moduli Didattici

1. **MODULO 1: Introduzione, Modelli Lineari & k-Nearest Neighbors**  
   * Slide: `ML-25-first-lectures-1-INTRO-v0.2`, `ML-25-linear-v0.1`, `ML-25-knn-v0.1`
2. **MODULO 2: Reti Neurali Multi-Layer (MLP) & Backpropagation**  
   * Slide: `ML-25-NN-part1-v.0.11`, `ML 24 Backpropagation notes latex v2 0`, `ML-25-NN-part2-v0.1`
3. **MODULO 3: Validazione, Model Selection & Assessment**  
   * Slide: `ML-25-Valid1-v.0.1`, `ML-25-Valid2-v0.1`, `ML-25-Valid3-v.0.1`
4. **MODULO 4: Statistical Learning Theory (SLT), Support Vector Machines & Bias-Varianza**  
   * Slide: `ML-25-SLT1-v.0.1`, `ML-25-SVM-v0.1`, `ML-25-SVM-other info-v.0.1`, `ML-25-Bias-Variance-v0.1`
5. **MODULO 5: Architetture Avanzate, Deep Learning, Apprendimento Non Supervisionato & Domini Strutturati**  
   * Slide: `ML-25-NN-part3-CNN-v0.1`, `ML-25-NN-part3-Deep-v0.1`, `ML-25-NN-part3-Rand-v0.1`, `ML-25-Unsupervised-SOM-v.0.1`, `ML-25-RNN-v0.1`, `ML-25-SDL-Intro-v0.1`

---

## 📌 MODULO 1: Introduzione, Modelli Lineari & k-NN

### 1.1 Introduzione & Framework di Machine Learning (`ML-25-first-lectures-1-INTRO-v0.2`)
- [ ] **Definizione di Machine Learning**: Approssimazione di funzioni incognite a partire da campioni dati osservati $D$.
- [ ] **Tassonomia dell'Apprendimento**:
  - *Supervisionato*: Classificazione Binaria/Multiclasse ($y \in \{-1, +1\}$ o $\{1\dots K\}$), Regressione Scalare/Multivariata ($y \in \mathbb{R}^K$).
  - *Non Supervisionato*: Clustering, Quantizzazione Vettoriale, Riduzione di Dimensionalità.
  - *Reinforcement Learning*: Apprendimento basato su reward/punizioni stocastiche.
- [ ] **Il Framework dei 4 Elementi di Ogni Algoritmo**:
  1. *I Dati ($D$)*: Dataset $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$, diviso in Train, Validation e Test set.
  2. *Lo Spazio delle Ipotesi ($\mathcal{H}$)*: Insieme delle funzioni $h: \mathcal{X} \to \mathcal{Y}$ rappresentabili dai pesi $w$.
  3. *La Loss Function ($\mathcal{L}$)*: Misura dell'errore (MSE, MEE 4D, BCE).
  4. *L'Algoritmo di Ottimizzazione*: Procedura per l'aggiornamento pesi (Gradient Descent, SGD).
- [ ] **Definizione Formale dei Rischi**:
  - *Rischio Generale / Errore Atteso*: $R(h) = \int_{\mathcal{X} \times \mathcal{Y}} L(h(x), y) \, dP(x, y)$
  - *Rischio Empirico*: $R_{emp}(h) = \frac{1}{N} \sum_{i=1}^N L(h(x_i), y_i)$
- [ ] **Loss Functions**:
  - *Mean Squared Error (MSE)*: $L_{MSE}(w) = \frac{1}{N} \sum_{i=1}^N (y_i - h(x_i))^2$
  - *Mean Euclidean Error (MEE 4D CUP)*: $L_{MEE}(w) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2 = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^4 (y_{i,m} - \hat{y}_{i,m})^2}$
  - *Binary Cross-Entropy (BCE)*: $L_{BCE}(w) = -\frac{1}{N} \sum_{i=1}^N \big[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \big]$
- [ ] **Algoritmo di Discesa del Gradiente**: $w^{(t+1)} = w^{(t)} - \eta \nabla L(w^{(t)})$
- [ ] **Inductive Bias (Bias Induttivo)**:
  - *Language / Restriction Bias*: Vincolo formale sulla classe di funzioni $\mathcal{H}$.
  - *Search / Preference Bias*: Criterio dell'ottimizzatore nella scelta tra le ipotesi in $\mathcal{H}$ (Occam's Razor).
  - *Unbiased Learner (Lookup Table)*: Senza bias induttivo non è possibile generalizzare su nuovi dati.

### 1.2 Modelli Lineari: Perceptron, LMS & Equazioni Normali (`ML-25-linear-v0.1`)
- [ ] **Il Perceptrone (Rosenblatt 1958)**:
  - Unità a Soglia Lineare (LTU): $f(x) = \text{sign}(w^T x + b) = \begin{cases} +1 & \text{se } w^T x + b \ge 0 \\ -1 & \text{se } w^T x + b < 0 \end{cases}$
  - Geometria dell'Iperpiano di Decisione: $w^T x + b = 0$, con vettore $w$ ortogonale all'iperpiano.
  - Regola di Aggiornamento On-Line Hebbiana: $w \leftarrow w + \eta (y_i - o_i) x_i$
  - *Teorema di Convergenza del Perceptrone (Novikoff 1962)*: Limite massimo sugli errori $k \le \frac{R^2}{\gamma^2}$ ($R = \max_i \|x_i\|$, margine $\gamma > 0$).
  - Esempio pratico porta NOT: $w = -2, b = +1 \implies f(x) = \text{sign}(-2x + 1)$.
  - Limite teorico: Impossibilità di separare l'XOR (Minsky & Papert 1969).
- [ ] **LMS (Least Mean Squares) / Regola del Delta (Widrow-Hoff)**:
  - Discesa continua prima della soglia su $net_i = w^T x_i + b$: $\Delta w = \eta (y_i - net_i) x_i$
  - Superficie d'errore parabolica ad unico minimo globale.
- [ ] **Soluzione in Forma Chiusa per la Regressione Lineare**:
  - *Equazioni Normali*: $w^* = (X^T X)^{-1} X^T y = X^+ y$
  - *Matrice Pseudoinversa di Moore-Penrose*: $X^+ = (X^T X)^{-1} X^T$
- [ ] **Linear Basis Expansion (LBE) & Teorema di Cover (1965)**:
  - Proiezione non lineare degli input $x \in \mathbb{R}^D$ in uno spazio di feature $\phi(x) \in \mathbb{R}^K$ ($K > D$):
    $$h(x) = w^T \phi(x) + b = \sum_{k=1}^K w_k \phi_k(x) + b$$
  - *Teorema di Cover (1965)*: La separabilità lineare diventa altamente probabile proiettando non linearmente i dati in uno spazio a dimensione elevata ($K \gg D$).

### 1.3 k-Nearest Neighbors (k-NN) & Curse of Dimensionality (`ML-25-knn-v0.1`)
- [ ] **Instance-Based / Lazy Learning**: Assenza di parametri liberi da addestrare.
- [ ] **Formulazione Formale di k-NN**:
  - *Regressione k-NN (Media dei Vicini)*: $\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$
  - *Classificazione k-NN (Voto a Maggioranza)*: $\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$
- [ ] **Maledizione della Dimensionalità (*Curse of Dimensionality*)**: Crollo della densità dei dati ed aumento del raggio dei vicinati ad alta dimensione.

---

## 📌 MODULO 2: Reti Neurali Multi-Layer (MLP) & Backpropagation

### 2.1 Funzioni di Attivazione e Proprietà (`ML-25-NN-part1-v.0.11`)
- [ ] **Sigmoide (Logistica)**: $\sigma(z) = \frac{1}{1+e^{-z}}$, derivata $\sigma'(z) = \sigma(z)(1-\sigma(z))$, saturazione per $|z|>4$ e Vanishing Gradient.
- [ ] **Tanh (Tangente Iperbolica)**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, derivata $\tanh'(z) = 1 - \tanh^2(z)$, proprietà zero-centered.
- [ ] **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$, derivata $f'(x) = \mathbb{I}(x > 0)$ (interruttore binario 0/1), derivata unitaria per $x>0$, Dying ReLU e **LeakyReLU** ($f(x) = \max(\alpha x, x)$).
- [ ] **ELU (Exponential Linear Unit)**: $f(x) = \begin{cases} x & \text{se } x > 0 \\ \alpha(e^x - 1) & \text{se } x \le 0 \end{cases}$, derivata $f'(x) = \begin{cases} 1 & \text{se } x > 0 \\ f(x) + \alpha & \text{se } x \le 0 \end{cases}$.
- [ ] **Dimostrazione della Necessità delle Attivazioni Non Lineari**: $o = W_L (W_{L-1} (\dots (W_1 x))) = W_{tot} x$ (composizione di mappe lineari = singolo modello lineare).

### 2.2 Architettura MLP & Teorema di Approssimazione Universale (`ML-25-NN-part1`)
- [ ] **Teorema di Approssimazione Universale (Hornik/Cybenko 1989)**:
  $$f(x) = \sum_{j=1}^M v_j \sigma(w_j^T x + b_j)$$
  Una rete feed-forward con 1 strato nascosto ed attivazioni non lineari può approssimare qualsiasi funzione continua su insiemi compatti con precisione $\epsilon > 0$.

### 2.3 L'Algoritmo di Backpropagation (`ML 24 Backpropagation notes latex v2 0`)
- [ ] **Forward Pass**: $net_j = \sum_u w_{ju} o_u$, $o_j = f_j(net_j)$.
- [ ] **Backward Pass (Chain Rule per i fattori di errore locale $\delta$)**:
  - *Neurone di Output $k$*: $\delta_k = (d_k - o_k) f'_k(net_k)$
  - *Neurone Nascosto $j$*: $\delta_j = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$
- [ ] **Regola di Aggiornamento Pesi**: $\frac{\partial E_p}{\partial w_{tu}} = -\delta_t o_u \implies \Delta w_{tu} = \eta \delta_t o_u$.
- [ ] **Efficienza Computazionale $O(|W|)$**: Fattorizzazione dei calcoli intermedi riutilizzando i $\delta$ del layer successivo.
- [ ] **Produttoria Chain Rule del Vanishing Gradient**:
  $$\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$$

### 2.4 Iperparametri & Addestramento (`ML-25-NN-part2-v0.1`)
- [ ] **Formula Integrale dell'Aggiornamento Pesi**:
  $$\Delta w_{tu}(t) = -\eta \frac{\partial E}{\partial w_{tu}(t)} + \alpha \Delta w_{tu}(t-1) - \eta \lambda w_{tu}(t)$$
- [ ] **Learning Rate ($\eta$)**: Tasso di apprendimento.
- [ ] **Momentum ($\alpha$)**: Inerzia fisica contro le oscillazioni (Nesterov NAG valuta il gradiente nel punto futuro $w + \alpha \Delta w_{old}$).
- [ ] **Weight Decay / Regolarizzazione $L_2$ di Tikhonov ($\lambda$)**: Penalità $\frac{\lambda}{2}\|w\|^2$ sulla norma dei pesi $\implies \Delta w \propto -\eta \lambda w$.
- [ ] **Inizializzazione dei Pesi**: Rottura della simmetria dei gradienti e prevenzione della saturazione iniziale.
- [ ] **Early Stopping**: Interruzione dell'addestramento al minimo della loss di validazione.
- [ ] **Ottimizzatori Avanzati**: R-Prop (Resilient Backprop), SGD, Mini-Batch, Adam, AdamW.

---

## 📌 MODULO 3: Validazione, Model Selection & Assessment

### 3.1 Schemi di Validazione Sperimentale (`ML-25-Valid1-v.0.1`)
- [ ] **Hold-Out Split**: Partizionamento semplice in Training Set, Validation Set e Test Set.
- [ ] **K-Fold Cross-Validation (`ML-25-Valid2-v0.1`)**:
  - Partizionamento in $K$ fold ruotando la validazione.
  - Stima della media e deviazione standard: $\bar{E}_{CV} = \frac{1}{K} \sum_{k=1}^K E_k, \quad \sigma_{CV} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (E_k - \bar{E}_{CV})^2}$.

### 3.2 Prevenzione del Data Leakage (`ML-25-Valid2-v0.1`)
- [ ] **Isolamento delle Trasformazioni (Scaling in-fold)**: Esecuzione del `fit_transform` dello scaler (`RobustScaler`/`StandardScaler`) ESCLUSIVAMENTE sul training fold di ciascun ciclo di Cross-Validation.

### 3.3 Model Selection, Assessment & Nested CV (`ML-25-Valid3-v.0.1`)
- [ ] **Model Selection**: Ricerca della migliore configurazione di iperparametri $h^*$ sul Validation Set.
- [ ] **Model Assessment**: Valutazione finale della capacità di generalizzazione su un Test Set indipendente.
- [ ] **Nested Cross-Validation (Doppio Ciclo)**:
  - *Inner Loop (K-Fold Interno)*: Model Selection (GridSearch / Optuna su $K-1$ fold).
  - *Outer Loop (K-Fold Esterno)*: Model Assessment (stima non polarizzata / *unbiased* su fold esterni non usati nella scelta degli iperparametri).

---

## 📌 MODULO 4: Statistical Learning Theory (SLT), Support Vector Machines & Bias-Varianza

### 4.1 Statistical Learning Theory (SLT) (`ML-25-SLT1-v.0.1`)
- [ ] **Shattering (Frammentazione)**: Capacità di realizzare tutte le $2^N$ dicotomie binaria (+1 / -1).
- [ ] **VC-Dimension ($h_{VC}$)**: MASSIMA cardinalità $N$ per cui esiste ALMENO UN insieme di $N$ punti shatterabile.
- [ ] **VC-Dimension degli Iperpiani Lineari ($VC = D + 1$)**: Dimostrazione formale tramite il **Teorema di Radon** ($\text{Conv}(A) \cap \text{Conv}(B) \neq \emptyset$).
- [ ] **Bound di Generalizzazione di Vapnik**:
  $$R(h) \le R_{emp}(h) + \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$
- [ ] **Structural Risk Minimization (SRM)**: Minimizzazione del bound complessivo lungo gerarchie annidate $\mathcal{H}_1 \subset \mathcal{H}_2 \dots \subset \mathcal{H}_k$.

### 4.2 Support Vector Machines (SVM & SVR) (`ML-25-SVM-v0.1` & `SVM-other info`)
- [ ] **Massimo Margine Geometrico**: $M = \frac{2}{\|w\|}$, con distanza ortogonale $d_i = \frac{|w^T x_i + b|}{\|w\|}$. Equivalenza con la minimizzazione di $\frac{1}{2} \|w\|^2$.
- [ ] **Hard Margin SVM Primale**: $\min_{w,b} \frac{1}{2} \|w\|^2 \quad \text{sotto vincoli } y_i(w^T x_i + b) \ge 1$.
- [ ] **Soft Margin SVM Primale**: $\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{sotto vincoli } y_i(w^T x_i + b) \ge 1 - \xi_i, \,\, \xi_i \ge 0$. ($C \to \infty$ Hard Margin vs $C$ piccolo Soft Margin).
- [ ] **Formulazione Duale & Condizioni KKT**:
  $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{sotto } 0 \le \alpha_i \le C, \,\, \sum_{i=1}^N \alpha_i y_i = 0$$
- [ ] **Sparsità KKT & Vettori di Supporto**: $\alpha_i \big[ y_i(w^T x_i + b) - 1 + \xi_i \big] = 0 \implies \alpha_i > 0$ solo per i Support Vector.
- [ ] **Funzione di Decisione Finale SVM**: $f(x) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right)$.
- [ ] **Kernel Trick & RBF Gaussiano**: $K(x, z) = \exp(-\gamma \|x-z\|^2)$ dove $\gamma = \frac{1}{2\sigma^2}$.
- [ ] **Support Vector Regression (SVR)**:
  - *Loss $\epsilon$-insensitive*: $L_\epsilon(y, f(x)) = \max(0, |y - f(x)| - \epsilon)$.
  - *Primale SVR con Slack Variables Doppie*: $\min \frac{1}{2} \|w\|^2 + C \sum (\xi_i + \xi_i^*)$ sotto vincoli $y_i - (w^T x_i + b) \le \epsilon + \xi_i, \, (w^T x_i + b) - y_i \le \epsilon + \xi_i^*$.

### 4.3 Decomposizione Bias-Varianza-Rumore (`ML-25-Bias-Variance-v0.1`)
- [ ] **Scomposizione Algebrica dell'Errore Quadratico Medio Atteso**:
  $$\mathbb{E}_{D,\epsilon}[(y - h_D(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D [ (h_D(x) - \bar{h}(x))^2 ]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$
- [ ] **Bias$^2$**: Errore di approssimazione del modello (**Underfitting**).
- [ ] **Varianza**: Sensibilità alle fluttuazioni del dataset di addestramento (**Overfitting**).
- [ ] **Rumore Irriducibile ($\sigma^2$)**: Incertezza stocastica intrinseca dei dati.

---

## 📌 MODULO 5: Architetture Avanzate, Deep Learning, Apprendimento Non Supervisionato & Domini Strutturati

### 5.1 Convolutional Neural Networks (CNN) (`ML-25-NN-part3-CNN-v0.1`)
- [ ] Feature maps, weight sharing, receptive field.
- [ ] Convoluzione Continua 1D: $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau$.
- [ ] Convoluzione Discreta 2D: $(I * K)[i, j] = \sum_m \sum_n I[i-m, j-n] K[m, n]$.
- [ ] **Cross-Correlazione 2D (PyTorch)**: $(I \star K)[i, j] = \sum_m \sum_n I[i+m, j+n] K[m, n]$.
- [ ] Parallelizzazione su GPU tramite l'operatore matriciale `im2col` (Image-to-Column).

### 5.2 Deep Learning & Risultati di Profondità (`ML-25-NN-part3-Deep-v0.1`)
- [ ] Produttoria della Chain Rule per il Vanishing Gradient: $\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$.
- [ ] Risultati di Profondità ("No-Flattening"): Vantaggio esponenziale delle architetture deep (parità a $N$-bit con $O(N)$ porte in $\log N$ strati vs $O(2^N)$ in 2 strati).
- [ ] Fenomeni Moderni: **Double Descent** (regime di interpolazione e sopra-parametrizzazione) e **Lottery Ticket Hypothesis** (Frankle & Carbin 2018).

### 5.3 Algoritmi Costruttivi & Random NN (`ML-25-NN-part3-Rand-v0.1`)
- [ ] **Cascade Correlation (Fahlman & Lebiere 1990)**:
  - Funzione obiettivo di covarianza: $S = \sum_{k \in \text{Output}} \left| \sum_p (o_p - \bar{o})(E_{p,k} - \bar{E}_k) \right|$.
  - Ascesa del gradiente col segno **PLUS**: $\Delta w_j = +\eta \frac{\partial S}{\partial w_j} = +\eta \sum_k \sigma_k \sum_p (E_{p,k} - \bar{E}_k) f'_p(net_c) x_{p,j}$.
  - Congelamento dei pesi d'ingresso per sempre (*frozen weights*) per risolvere il *moving target problem*.
- [ ] **Random Neural Networks & Reservoir Computing (Echo State Networks ESN)**:
  - Transizione di stato del bacino: $x(t+1) = \tanh(W_{in} u(t+1) + W_{res} x(t))$.
  - Output di Readout: $y(t+1) = W_{out} x(t+1)$.
  - Pesi $W_{res}$ congelati casualmente; addestramento analitico chiuso di $W_{out}$ tramite regressione lineare.

### 5.4 Apprendimento Non Supervisionato & SOM (`ML-25-Unsupervised-SOM-v.0.1`)
- [ ] **Quantizzazione Vettoriale & K-Means**:
  - Celle di Voronoi: $V_i = \{x \in \mathcal{X} : \|x - \mu_i\| \le \|x - \mu_j\| \,\, \forall j \neq i\}$.
  - Errore di Distorsione Globale: $E = \sum_{i=1}^k \sum_{x \in V_i} \|x - \mu_i\|^2$ (discreto) e $E = \sum_{i=1}^k \int_{V_i} \|x - \mu_i\|^2 p(x) dx$ (continuo).
  - Regola On-line K-means: $\mu_i \leftarrow \mu_i + \eta (x - \mu_i)$.
- [ ] **Self-Organizing Maps (SOM di Kohonen 1982)**:
  - Mappa topologica bidimensionale.
  - Ricerca della Best Matching Unit (BMU): $\text{BMU} = \arg\min_i \|x - w_i\|$.
  - Aggiornamento pesi: $w_i(t+1) = w_i(t) + \eta(t) h_{ic}(t) (x(t) - w_i(t))$ con vicinato gaussiano $h_{ic}(t) = \exp\left(-\frac{\|r_i - r_c\|^2}{2\sigma(t)^2}\right)$.
- [ ] **Autoencoders (Undercomplete vs Overcomplete)**:
  - Ricostruzione: $\|x - \hat{x}\|^2 = \|x - f(g(x))\|^2$.
  - Undercomplete ($dim(z) < dim(x)$) per compressione non lineare latente.
  - Overcomplete ($dim(z) > dim(x)$) con vincoli di Sparsity o Denoising per il pre-training.
- [ ] **Ensemble Learning (Voting, Bagging, Boosting)**:
  - Schema Voto/Media del Comitato: $h_{ens}(x) = \frac{1}{M} \sum h_m(x)$ (regressione), $h_{ens}(x) = \arg\max_c \sum \mathbb{I}(h_m(x)=c)$ (classificazione).
  - Bagging (Bootstrap Aggregation) per modelli ad alta varianza (averaging).
  - Boosting (AdaBoost) per modelli ad alto bias (ripesaggio sequenziale degli errori).

### 5.5 Recurrent Neural Networks (RNN) (`ML-25-RNN-v0.1`)
- [ ] Transizione di Stato: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
- [ ] Output del Layer: $y_t = \text{softmax}(W_{hy} h_t + b_y)$.
- [ ] Backpropagation Through Time (BPTT).
- [ ] **Gradient Clipping**: $g \leftarrow v \frac{g}{\|g\|_2}$ se $\|g\|_2 > v$ per evitare l'esplosione dei gradienti.

### 5.6 Domini Strutturati & Graph Neural Networks (`ML-25-SDL-Intro-v0.1`)
- [ ] **Message Passing per GNN**:
  - `AGGREGATE`: $m_v^{(k)} = \text{AGGREGATE}\left(\{h_u^{(k-1)} : u \in N(v)\}\right)$.
  - `UPDATE`: $h_v^{(k)} = \text{UPDATE}\left(h_v^{(k-1)}, m_v^{(k)}\right)$.
- [ ] **NN4G (Neural Network for Graphs)**: Formulazione ricorsiva $h_v^{(k)} = f\left( W_1 x_v + \sum W_{2,l} \sum_{u \in N(v)} h_u^{(l)} \right)$.
- [ ] **Readout Globale su Grafi**: $y_G = \text{Readout}\left( \sum_{v \in V} h_v \right) = W_{out} \left( \sum_{v \in V} h_v \right)$.
- [ ] **Word Embeddings**: Rappresentazioni distribuite dense e relazioni geometrico-semantiche ($\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$).


---

## ✍️ SEZIONE DEDICATA: Le 8 Dimostrazioni Matematiche Fondamentali della Prova Scritta

Questa sezione raccoglie le **8 dimostrazioni algebrico-matematiche trasversali** maggiormente richieste nella prova scritta pre-orale dal Prof. Micheli, spiegate passo-passo con tutti i passaggi intermedi:

### 📝 1. Dimostrazione analitica delle Equazioni Normali (Regressione Lineare)
* **Obiettivo**: Minimizzare la Loss MSE in forma matriciale $L(w) = \frac{1}{N} \|X w - y\|_2^2$.
* **Sviluppo Matematico**:
  $$L(w) = \frac{1}{N} (X w - y)^T (X w - y) = \frac{1}{N} \left( w^T X^T X w - 2 w^T X^T y + y^T y \right)$$
* **Calcolo del Gradiente**:
  $$\nabla_w L(w) = \frac{2}{N} \left( X^T X w - X^T y \right)$$
* **Annullamento del Gradiente**:
  $$\nabla_w L(w) = 0 \implies X^T X w = X^T y \implies w^* = (X^T X)^{-1} X^T y = X^+ y$$
  dove $X^+$ è la Pseudoinversa di Moore-Penrose.

---

### 📝 2. Teorema di Convergenza del Perceptrone (Novikoff 1962)
* **Assunzione di Separabilità**: Esiste un vettore ottimo $w^*$ con $\|w^*\|=1$ tale che $y_i (w^{*T} x_i) \ge \gamma > 0$ per ogni pattern $i$.
* **Limite Superiore su $\|w_k\|^2$**: Ad ogni errore si applica l'aggiornamento $w_{k} = w_{k-1} + \eta y_i x_i$:
  $$\|w_k\|^2 = \|w_{k-1} + \eta y_i x_i\|^2 = \|w_{k-1}\|^2 + 2 \eta \underbrace{y_i (w_{k-1}^T x_i)}_{\le 0 \text{ (errore)}} + \eta^2 \|x_i\|^2 \le \|w_{k-1}\|^2 + \eta^2 R^2$$
  Ricorsivamente su $k$ errori: $\|w_k\|^2 \le k \eta^2 R^2$ (dove $R = \max_i \|x_i\|$).
* **Limite Inferiore su $w_k^T w^*$**:
  $$w_k^T w^* = (w_{k-1} + \eta y_i x_i)^T w^* = w_{k-1}^T w^* + \eta \underbrace{y_i (w^{*T} x_i)}_{\ge \gamma} \ge w_{k-1}^T w^* + \eta \gamma$$
  Ricorsivamente su $k$ errori: $w_k^T w^* \ge k \eta \gamma$.
* **Disuguaglianza di Cauchy-Schwarz**:
  $$(k \eta \gamma)^2 \le (w_k^T w^*)^2 \le \|w_k\|^2 \|w^*\|^2 \le k \eta^2 R^2 \cdot 1 \implies k^2 \eta^2 \gamma^2 \le k \eta^2 R^2 \implies k \le \frac{R^2}{\gamma^2}$$

---

### 📝 3. Derivazione Completa dell'Algoritmo di Backpropagation
* **Forward Pass**: $net_j = \sum_u w_{ju} o_u$, $o_j = f_j(net_j)$.
* **Derivata del Peso via Chain Rule**: $\frac{\partial E_p}{\partial w_{tu}} = \frac{\partial E_p}{\partial net_t} \cdot \frac{\partial net_t}{\partial w_{tu}}$.
  Poiché $\frac{\partial net_t}{\partial w_{tu}} = o_u$ e definendo $\delta_t = -\frac{\partial E_p}{\partial net_t}$, si ottiene:
  $$\frac{\partial E_p}{\partial w_{tu}} = -\delta_t o_u \implies \Delta w_{tu} = \eta \delta_t o_u$$
* **Fattore d'Errore $\delta_k$ (Neurone di Output)**:
  $$\delta_k = -\frac{\partial E_p}{\partial net_k} = -\frac{\partial E_p}{\partial o_k} \cdot \frac{\partial o_k}{\partial net_k} = (d_k - o_k) f'_k(net_k)$$
* **Fattore d'Errore $\delta_j$ (Neurone Nascosto)**:
  $$\delta_j = -\frac{\partial E_p}{\partial net_j} = -\sum_{k \in \text{Output}} \frac{\partial E_p}{\partial net_k} \frac{\partial net_k}{\partial o_j} \frac{\partial o_j}{\partial net_j} = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$$

---

### 4. Decomposizione Algebrica Bias-Varianza-Rumore
* **Sviluppo dell'Errore Atteso**: Sia $y = f(x) + \epsilon$ con $\mathbb{E}[\epsilon]=0$ e $\text{Var}(\epsilon)=\sigma^2$.
  $$\mathbb{E}_{D,\epsilon}[(y - h_D(x))^2] = \mathbb{E}_{D,\epsilon}[(f(x) + \epsilon - h_D(x))^2] = \mathbb{E}_D[(f(x) - h_D(x))^2] + \sigma^2$$
* **Aggiunta e Sottrazione di $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$**:
  $$\mathbb{E}_D[(f(x) - \bar{h}(x) + \bar{h}(x) - h_D(x))^2] = \mathbb{E}_D[(f(x) - \bar{h}(x))^2] + \mathbb{E}_D[(\bar{h}(x) - h_D(x))^2] + 2(f(x) - \bar{h}(x)) \underbrace{\mathbb{E}_D[\bar{h}(x) - h_D(x)]}_{=0}$$
* **Risultato Finale**:
  $$\mathbb{E}[(y - h_D(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D[(h_D(x) - \bar{h}(x))^2]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$

---

### 📝 5. Teorema di Radon & VC-Dimension degli Iperpiani ($VC = D + 1$)
* **Teorema di Radon**: Un qualsiasi insieme di $D + 2$ punti in $\mathbb{R}^D$ può essere partizionato in due insiemi disgiunti $A$ e $B$ tali che le loro inviluppi convesse si intersecano: $\text{Conv}(A) \cap \text{Conv}(B) \neq \emptyset$.
* **Impossibilità di Shatterare $D+2$ punti**: Sia $x^* \in \text{Conv}(A) \cap \text{Conv}(B)$. Se un iperpiano lineare assegna etichetta $+1$ a tutti i punti di $A$, per convessità deve assegnare $+1$ anche a $x^*$. Ma se assegna $-1$ a tutti i punti di $B$, per convessità deve assegnare $-1$ anche a $x^*$. Si genera una contraddizione ($x^*$ non può avere etichetta $+1$ e $-1$ simultaneamente).
* **Conclusione**: Nessun iperpiano lineare in $\mathbb{R}^D$ può shatterare $D+2$ punti $\implies VC = D + 1$.

---

### 📝 6. Derivazione Primal-Dual per le Support Vector Machines (SVM)
* **Problema Primale (Soft Margin)**:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t. } y_i(w^T x_i + b) \ge 1 - \xi_i, \,\, \xi_i \ge 0$$
* **Lagrangiana**:
  $$L(w, b, \xi, \alpha, \mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i \big( y_i(w^T x_i + b) - 1 + \xi_i \big) - \sum_{i=1}^N \mu_i \xi_i$$
* **Condizioni di Stazionarietà KKT**:
  1. $\nabla_w L = 0 \implies w = \sum_{i=1}^N \alpha_i y_i x_i$
  2. $\frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$
  3. $\frac{\partial L}{\partial \xi_i} = 0 \implies C - \alpha_i - \mu_i = 0 \implies 0 \le \alpha_i \le C$
* **Sostituzione nella Lagrangiana (Formulazione Duale)**:
  $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t. } 0 \le \alpha_i \le C, \,\, \sum_{i=1}^N \alpha_i y_i = 0$$

---

### 📝 7. Bound di Vapnik & Principio di Minimizzazione del Rischio Strutturale (SRM)
* **Bound di Generalizzazione della SLT**: Per qualsiasi ipotesi $h \in \mathcal{H}$ con dimensione VC $h_{VC}$, con probabilità almeno $1-\eta$:
  $$R(h) \le R_{emp}(h) + \underbrace{\sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}}_{\text{Confidenza VC }\Omega(N, h_{VC})}$$
* **Principio SRM**: Struttura $\mathcal{H}$ in una gerarchia annidata di spazi a complessità crescente $\mathcal{H}_1 \subset \mathcal{H}_2 \dots \subset \mathcal{H}_k$ con $h_{VC}^{(1)} < h_{VC}^{(2)} \dots$. Seleziona l'ipotesi $h^*$ che minimizza la somma $R_{emp}(h) + \Omega(N, h_{VC})$, bilanciando fitting ed overfitting.

---

### 📝 8. Cascade Correlation: Massimizzazione della Covarianza $S$
* **Funzione Obiettivo del Candidato $c$**:
  $$S = \sum_{k \in \text{Output}} \left| \sum_p (o_{p,c} - \bar{o}_c)(E_{p,k} - \bar{E}_k) \right|$$
  dove $o_{p,c}$ è l'uscita del candidato sul pattern $p$, $\bar{o}_c$ è la sua media, $E_{p,k}$ è l'errore del neurone di output $k$ e $\bar{E}_k$ è l'errore medio su $k$.
* **Ascesa del Gradiente (Sign PLUS)**:
  $$\Delta w_{c,j} = +\eta \frac{\partial S}{\partial w_{c,j}} = +\eta \sum_{k} \sigma_k \sum_p (E_{p,k} - \bar{E}_k) f'_p(net_c) x_{p,j}$$
  dove $\sigma_k = \text{sign}\left( \sum_p (o_{p,c} - \bar{o}_c)(E_{p,k} - \bar{E}_k) \right)$.
* **Principio dei Pesi Congelati (*Frozen Weights*)**: Una volta scelto il candidato migliore e connesso alla rete, i suoi pesi d'ingresso $w_{c,j}$ vengono **congelati permanentemente**, evitando il problema del *moving target*.
