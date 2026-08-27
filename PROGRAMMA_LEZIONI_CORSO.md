# 📚 Programma e Indice Strutturato delle Lezioni: Machine Learning
**Corso del Prof. Alessio Micheli — Università di Pisa**  
*(Elenco Ordinato e Strutturato degli Argomenti Trattati a Lezione — Mappatura 1:1 sui File delle Slide [ML-25])*

---

## 🗺️ Struttura dei Modelli Didattici

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
- [ ] **Definizione di Machine Learning**: Approssimazione di funzioni incognite a partire da campioni dati osservati.
- [ ] **Tassonomia dell'Apprendimento**:
  - Apprendimento Supervisionato (Classificazione Binaria/Multiclasse, Regressione Scalare/Multivariata).
  - Apprendimento Non Supervisionato (Clustering, Quantizzazione Vettoriale, Riduzione di Dimensionalità).
  - Apprendimento per Rinforzo (Reinforcement Learning).
- [ ] **Il Framework dei 4 Elementi di Ogni Algoritmo**:
  1. *I Dati ($D$)*: Dataset di addestramento, validazione e test $D = \{(x_i, y_i)\}_{i=1}^N$.
  2. *Lo Spazio delle Ipotesi ($\mathcal{H}$)*: Insieme delle funzioni ammissibili dal modello.
  3. *La Loss Function ($\mathcal{L}$)*: Misura dell'errore (MSE, MEE 4D per la CUP, BCE).
  4. *L'Algoritmo di Ottimizzazione*: Algoritmo numerico per l'aggiornamento pesi (Gradient Descent, SGD).
- [ ] **Definizione Formale dei Rischi**:
  - *Rischio Generale / Errore Atteso ($R(h)$)*: $R(h) = \int_{\mathcal{X} \times \mathcal{Y}} L(h(x), y) \, dP(x, y)$ (non direttamente calcolabile).
  - *Rischio Empirico ($R_{emp}(h)$)*: $R_{emp}(h) = \frac{1}{N} \sum_{i=1}^N L(h(x_i), y_i)$ (calcolato sul dataset osservato).
- [ ] **Inductive Bias (Bias Induttivo)**:
  - *Language / Restriction Bias*: Limitazione formale della classe di funzioni in $\mathcal{H}$.
  - *Search / Preference Bias*: Preferenza accordata dall'ottimizzatore a determinate ipotesi in $\mathcal{H}$.
  - *Unbiased Learner (Lookup Table)*: L'impossibilità di generalizzare su dati non visti in assenza di bias induttivo.

### 1.2 Modelli Lineari: Perceptron, LMS & Equazioni Normali (`ML-25-linear-v0.1`)
- [ ] **Il Perceptrone (Rosenblatt 1958)**:
  - Unità a Soglia Lineare (LTU): $f(x) = \text{sign}(w^T x + b)$.
  - Geometria dell'Iperpiano di Decisione: $w^T x + b = 0$, vettori ortogonali.
  - Regola di Aggiornamento On-Line Hebbiana: $w \leftarrow w + \eta (y_i - o_i) x_i$.
  - *Teorema di Convergenza del Perceptrone (Novikoff 1962)*: Limite massimo sugli errori $k \le \frac{R^2}{\gamma^2}$ sotto separabilità lineare con margine $\gamma > 0$.
  - Esempio pratico: Realizzazione della porta logica NOT ($w=-2, b=+1$).
  - Limite teorico: Impossibilità di separare l'XOR (Minsky & Papert 1969).
- [ ] **LMS (Least Mean Squares) / Regola del Delta (Widrow-Hoff)**:
  - Discesa del gradiente continua prima della soglia: $\Delta w = \eta (y_i - net_i) x_i$.
  - Superficie d'errore parabolica quadratica ad unico minimo globale.
- [ ] **Soluzione in Forma Chiusa per la Regressione Lineare**:
  - *Equazioni Normali*: $w^* = (X^T X)^{-1} X^T y = X^+ y$.
  - *Matrice Pseudoinversa di Moore-Penrose*: $X^+ = (X^T X)^{-1} X^T$.
- [ ] **Linear Basis Expansion (LBE) & Teorema di Cover (1965)**:
  - Proiezione non lineare degli input $x \in \mathbb{R}^D$ in uno spazio di feature $\phi(x) \in \mathbb{R}^K$ ($K > D$).
  - *Teorema di Cover (1965)*: La separabilità lineare diventa altamente probabile proiettando non linearmente i dati in spazi a dimensione elevata.

### 1.3 k-Nearest Neighbors (k-NN) & Curse of Dimensionality (`ML-25-knn-v0.1`)
- [ ] **Instance-Based / Lazy Learning**: Assenza di parametri liberi da addestrare in train.
- [ ] **Formulazione Formale di k-NN**:
  - *Regressione k-NN (Media dei Vicini)*: $\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$.
  - *Classificazione k-NN (Voto a Maggioranza)*: $\hat{y}(x) = \arg\max_c \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$.
- [ ] **Maledizione della Dimensionalità (*Curse of Dimensionality*)**: Perdita di località dei vicinati e degradamento delle metriche di similarità al crescere delle dimensioni $D$.

---

## 📌 MODULO 2: Reti Neurali Multi-Layer (MLP) & Backpropagation

### 2.1 Funzioni di Attivazione e Proprietà (`ML-25-NN-part1-v.0.11`)
- [ ] **Sigmoide (Logistica)**: $\sigma(z) = \frac{1}{1+e^{-z}}$, derivata $\sigma'(z) = \sigma(z)(1-\sigma(z))$, problema della saturazione e Vanishing Gradient.
- [ ] **Tanh (Tangente Iperbolica)**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, derivata $\tanh'(z) = 1 - \tanh^2(z)$, proprietà zero-centered.
- [ ] **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$, derivata $f'(x) = \mathbb{I}(x > 0)$ (interruttore binario 0/1), risoluzione del Vanishing Gradient per $x>0$, Dying ReLU e **LeakyReLU** ($f(x) = \max(\alpha x, x)$).
- [ ] **ELU (Exponential Linear Unit)**: $f(x) = x$ per $x>0$ e $\alpha(e^x-1)$ per $x \le 0$, transizione fluida e media bilanciata a zero.
- [ ] **Dimostrazione della Necessità delle Attivazioni Non Lineari**: La composizione di mappe lineari crolla algebricamente in un unico modello lineare $o = W_{tot} x$.

### 2.2 Architettura MLP & Teorema di Approssimazione Universale (`ML-25-NN-part1`)
- [ ] Struttura a strati: Input layer, Hidden layers, Output layer.
- [ ] **Teorema di Approssimazione Universale (Hornik/Cybenko 1989)**: Capacità di approssimare qualsiasi funzione continua su insiemi compatti con 1 strato nascosto ed attivazioni non lineari.

### 2.3 L'Algoritmo di Backpropagation (`ML 24 Backpropagation notes latex v2 0`)
- [ ] **Forward Pass**: $net_j = \sum_u w_{ju} o_u$, $o_j = f_j(net_j)$.
- [ ] **Backward Pass (Chain Rule per i fattori di errore locale $\delta$)**:
  - *Neurone di Output $k$*: $\delta_k = (d_k - o_k) f'_k(net_k)$.
  - *Neurone Nascosto $j$*: $\delta_j = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$.
- [ ] **Regola di Aggiornamento Pesi**: $\Delta w_{tu} = \eta \delta_t o_u$.
- [ ] **Efficienza Computazionale**: Riduzione della complessità a $O(|W|)$ tramite riutilizzo dei $\delta$ calcolati al layer successivo.

### 2.4 Iperparametri & Addestramento (`ML-25-NN-part2-v0.1`)
- [ ] **Formula Integrale dell'Aggiornamento Pesi**:
  $$\Delta w_{tu}(t) = -\eta \frac{\partial E}{\partial w_{tu}(t)} + \alpha \Delta w_{tu}(t-1) - \eta \lambda w_{tu}(t)$$
- [ ] **Learning Rate ($\eta$)**: Impatto sulla convergenza e stabilità della discesa.
- [ ] **Momentum ($\alpha$)**: Inerzia fisica e smorzamento delle oscillazioni nei canyon (Nesterov Accelerated Gradient NAG).
- [ ] **Weight Decay / Regolarizzazione $L_2$ di Tikhonov ($\lambda$)**: Controllo della norma dei pesi $\frac{\lambda}{2}\|w\|^2$.
- [ ] **Inizializzazione dei Pesi**: Rottura della simmetria dei gradienti e prevenzione della saturazione iniziale.
- [ ] **Early Stopping**: Interruzione dell'addestramento basata sul Validation set.
- [ ] **Ottimizzatori Avanzati**: R-Prop (Resilient Backprop), SGD, Mini-Batch, Adam, AdamW.

---

## 📌 MODULO 3: Validazione, Model Selection & Assessment

### 3.1 Schemi di Validazione Sperimentale (`ML-25-Valid1-v.0.1`)
- [ ] **Hold-Out Split**: Partizionamento semplice in Training Set, Validation Set e Test Set.
- [ ] **K-Fold Cross-Validation (`ML-25-Valid2-v0.1`)**:
  - Partizionamento in $K$ fold ruotando la validazione.
  - Stima della media e deviazione standard: $\bar{E}_{CV} \pm \sigma_{CV}$.

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
- [ ] **Shattering (Frammentazione)**: Capacità di un'ipotesi $h \in \mathcal{H}$ di realizzare tutte le $2^N$ dicotomie binaria (+1 / -1).
- [ ] **VC-Dimension ($h_{VC}$)**: MASSIMA cardinalità $N$ per cui esiste ALMENO UN insieme di $N$ punti shatterabile.
- [ ] **VC-Dimension degli Iperpiani Lineari ($VC = D + 1$)**: Dimostrazione formale tramite il **Teorema di Radon** ($\text{Conv}(A) \cap \text{Conv}(B) \neq \emptyset$).
- [ ] **Bound di Generalizzazione di Vapnik**:
  $$R(h) \le R_{emp}(h) + \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$
- [ ] **Structural Risk Minimization (SRM)**: Minimizzazione del bound complessivo lungo gerarchie annidate $\mathcal{H}_1 \subset \mathcal{H}_2 \dots \subset \mathcal{H}_k$.

### 4.2 Support Vector Machines (SVM & SVR) (`ML-25-SVM-v0.1` & `SVM-other info`)
- [ ] **Massimo Margine Geometrico**: $M = \frac{2}{\|w\|}$, equivalenza con la minimizzazione di $\frac{1}{2} \|w\|^2$.
- [ ] **Soft Margin SVM**: Slack variables $\xi_i \ge 0$, parametro di regolarizzazione $C$ ($C \to \infty$ Hard Margin vs $C$ piccolo Soft Margin).
- [ ] **Formulazione Duale & Condizioni KKT**: Moltiplicatori di Lagrange $\alpha_i$, Vettori di Supporto ($\alpha_i > 0$).
- [ ] **Kernel Trick & Teorema di Mercer**: Matrice di Gram semidefinita positiva, **Kernel RBF Gaussiano** $K(x, z) = \exp(-\gamma \|x-z\|^2)$ ed iperparametro $\gamma$.
- [ ] **Support Vector Regression (SVR)**: Loss $\epsilon$-insensitive $L_\epsilon(y, f(x)) = \max(0, |y - f(x)| - \epsilon)$, tubo di tolleranza $\pm \epsilon$, slack variables doppie $(\xi_i, \xi_i^*)$.

### 4.3 Decomposizione Bias-Varianza-Rumore (`ML-25-Bias-Variance-v0.1`)
- [ ] **Scomposizione Algebrica dell'Errore Quadratico Medio Atteso**:
  $$\mathbb{E}[(y - h_D(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D [ (h_D(x) - \bar{h}(x))^2 ]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$
- [ ] **Bias$^2$**: Errore di approssimazione del modello (**Underfitting**).
- [ ] **Varianza**: Sensibilità alle fluttuazioni del dataset di addestramento (**Overfitting**).
- [ ] **Rumore Irriducibile ($\sigma^2$)**: Incertezza stocastica intrinseca dei dati.

---

## 📌 MODULO 5: Architetture Avanzate, Deep Learning, Apprendimento Non Supervisionato & Domini Strutturati

### 5.1 Convolutional Neural Networks (CNN) (`ML-25-NN-part3-CNN-v0.1`)
- [ ] Feature maps, weight sharing, receptive field.
- [ ] Convoluzione Discreta 2D vs **Cross-Correlazione 2D** in PyTorch: $(I \star K)[i, j] = \sum_m \sum_n I[i+m, j+n] K[m, n]$.
- [ ] Parallelizzazione su GPU tramite l'operatore matriciale `im2col` (Image-to-Column).

### 5.2 Deep Learning & Risultati di Profondità (`ML-25-NN-part3-Deep-v0.1`)
- [ ] Produttoria della Chain Rule per il Vanishing Gradient: $\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$.
- [ ] Risultati di Profondità ("No-Flattening"): Vantaggio esponenziale delle architetture deep (parità a $N$-bit con $O(N)$ porte in $\log N$ strati vs $O(2^N)$ in 2 strati).
- [ ] Fenomeni Moderni: **Double Descent** (regime di interpolazione e sopra-parametrizzazione) e **Lottery Ticket Hypothesis** (Frankle & Carbin 2018).

### 5.3 Algoritmi Costruttivi & Random NN (`ML-25-NN-part3-Rand-v0.1`)
- [ ] **Cascade Correlation (Fahlman & Lebiere 1990)**:
  - Generazione di neuroni candidati ed addestramento per **massimizzare la covarianza $S$** con l'errore residuo della rete.
  - Ascesa del gradiente col segno **PLUS** ($+\eta \frac{\partial S}{\partial w}$).
  - Congelamento dei pesi d'ingresso per sempre (*frozen weights*) per risolvere il *moving target problem*.
- [ ] **Random Neural Networks & Reservoir Computing (Echo State Networks ESN)**:
  - Pesi dello strato nascosto ($W_{res}$) casuali e congelati permanently.
  - Addestramento del solo readout layer in forma analitica chiusa tramite regressione lineare.

### 5.4 Apprendimento Non Supervisionato & SOM (`ML-25-Unsupervised-SOM-v.0.1`)
- [ ] **Quantizzazione Vettoriale & K-Means**:
  - Celle di Voronoi: $V_i = \{x \in \mathcal{X} : \|x - \mu_i\| \le \|x - \mu_j\| \,\, \forall j \neq i\}$.
  - Errore di Distorsione Globale: $E = \sum_{i=1}^k \sum_{x \in V_i} \|x - \mu_i\|^2$.
  - Regola On-line K-means: $\mu_i \leftarrow \mu_i + \eta (x - \mu_i)$.
- [ ] **Self-Organizing Maps (SOM di Kohonen 1982)**:
  - Mappa topologica bidimensionale.
  - Ricerca della Best Matching Unit (BMU) ed aggiornamento vicinato gaussiano $h_{ic}(t) = \exp\left(-\frac{\|r_i - r_c\|^2}{2\sigma(t)^2}\right)$.
- [ ] **Autoencoders (Undercomplete vs Overcomplete)**:
  - Undercomplete ($dim(z) < dim(x)$) per compressione non lineare latente.
  - Overcomplete ($dim(z) > dim(x)$) con vincoli di Sparsity o Denoising per il pre-training.
- [ ] **Ensemble Learning (Voting, Bagging, Boosting)**:
  - Schema Voto/Media del Comitato: $h_{ens}(x) = \frac{1}{M} \sum h_m(x)$.
  - Bagging (Bootstrap Aggregation) per modelli ad alta varianza (averaging).
  - Boosting (AdaBoost) per modelli ad alto bias (ripesaggio sequenziale degli errori).

### 5.5 Recurrent Neural Networks (RNN) (`ML-25-RNN-v0.1`)
- [ ] Modellazione di sequenze temporali con memoria interna.
- [ ] Transizione di Stato: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
- [ ] Backpropagation Through Time (BPTT).
- [ ] **Gradient Clipping**: $g \leftarrow v \frac{g}{\|g\|_2}$ se $\|g\|_2 > v$ per evitare l'esplosione dei gradienti.

### 5.6 Domini Strutturati & Graph Neural Networks (`ML-25-SDL-Intro-v0.1`)
- [ ] **Message Passing per GNN**:
  - `AGGREGATE`: $m_v^{(k)} = \text{AGGREGATE}\left(\{h_u^{(k-1)} : u \in N(v)\}\right)$.
  - `UPDATE`: $h_v^{(k)} = \text{UPDATE}\left(h_v^{(k-1)}, m_v^{(k)}\right)$.
- [ ] **NN4G (Neural Network for Graphs)**: Formulazione ricorsiva $h_v^{(k)} = f\left( W_1 x_v + \sum W_{2,l} \sum_{u \in N(v)} h_u^{(l)} \right)$.
- [ ] **Readout Globale su Grafi**: $y_G = W_{out} \left( \sum_{v \in V} h_v \right)$.
- [ ] **Word Embeddings**: Rappresentazioni distribuite dense e relazioni geometrico-semantiche ($\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$).
