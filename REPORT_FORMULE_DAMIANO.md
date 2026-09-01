# 📐 Report Formule, Dimostrazioni e Simboli (Riordinato)
**Autore Originale: Damiano Degliotti — Corso del Prof. Alessio Micheli**  
*(Formulario Integrale Re-strutturato in Ordine Cronologico 1:1 sulle Slide Ufficiali del Corso [ML-25])*

---

## 📖 Legenda dei Simboli Principali

* $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$: Dataset osservato composto da $N$ campioni.
* $x_i \in \mathbb{R}^D$: Vettore delle feature d'ingresso a $D$ dimensioni per il pattern $i$.
* $y_i \in \mathcal{Y}$: Target/etichetta reale per il pattern $i$ (scalare per regressione 1D, vettoriale $\mathbb{R}^K$ per CUP, binario $\{-1, +1\}$ o multiclasse $\{1\dots C\}$).
* $h(x) \in \mathcal{H}$: Ipotesi/predizione generata dal modello all'interno dello Spazio delle Ipotesi $\mathcal{H}$.
* $w \in \mathbb{R}^D$: Vettore dei pesi dell'iperpiano o della rete neurale.
* $b \in \mathbb{R}$: Termine di bias/soglia dell'iperpiano o della rete.
* $L(h(x), y)$: Loss function (funzione di costo istantanea su un singolo pattern).
* $R(h)$: Rischio Generale / Errore Atteso teorico sull'intera distribuzione congiunta $P(x,y)$.
* $R_{emp}(h)$: Rischio Empirico / Errore medio calcolato sul dataset osservato.
* $\eta > 0$: Learning Rate (tasso di apprendimento per la discesa del gradiente).
* $\alpha \in [0, 1)$: Coefficiente di Momentum per la discesa del gradiente.
* $\lambda \ge 0$: Parametro di Weight Decay (regolarizzazione $L_2$ di Tikhonov).
* $\gamma > 0$: Margine geometrico del Perceptrone / Iperparametro della gaussiana RBF nelle SVM.
* $\xi_i \ge 0$: Variabili Slack nella Soft Margin SVM per gestire i punti dentro il margine o errati.
* $\alpha_i \ge 0$: Moltiplicatori di Lagrange duali nella formulazione SVM.
* $h_{VC}$: Dimensione Vapnik-Chervonenkis (VC-Dimension), misura della capacità espressiva dello spazio $\mathcal{H}$.
* $S$: Funzione obiettivo di covarianza nell'algoritmo Cascade Correlation.
* $V_i$: Cella di Voronoi del centroide $\mu_i$ nella Quantizzazione Vettoriale / K-Means.
* $\text{BMU}$: Best Matching Unit nelle Mappe Auto-Organizzanti (SOM di Kohonen).

---

## 📌 MODULO 1: Introduzione, Modelli Lineari & k-NN (Slides `INTRO`, `linear`, `knn`)

### 1.1 Framework di Machine Learning & Rischi (`ML-25-first-lectures-1-INTRO-v0.2`)
* **Rischio Generale / Errore Atteso ($R(h)$)**:
  $$R(h) = \int_{\mathcal{X} \times \mathcal{Y}} L(h(x), y) \, dP(x, y)$$
* **Rischio Empirico ($R_{emp}(h)$)**:
  $$R_{emp}(h) = \frac{1}{N} \sum_{i=1}^N L(h(x_i), y_i)$$
* **Mean Squared Error (MSE)**:
  $$L_{MSE}(w) = \frac{1}{N} \sum_{i=1}^N (y_i - h(x_i))^2$$
* **Mean Euclidean Error (MEE 4D ML CUP)**:
  $$L_{MEE}(w) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2 = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^4 (y_{i,m} - \hat{y}_{i,m})^2}$$
* **Binary Cross-Entropy (BCE)**:
  $$L_{BCE}(w) = -\frac{1}{N} \sum_{i=1}^N \big[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \big]$$
* **Discesa del Gradiente Generica**:
  $$w^{(t+1)} = w^{(t)} - \eta \nabla L(w^{(t)})$$

---

### 1.2 Modelli Lineari: Perceptron, LMS & Equazioni Normali (`ML-25-linear-v0.1`)

#### A) Il Perceptrone (Rosenblatt 1958)
* **Formulazione LTU**:
  $$f(x) = \text{sign}(w^T x + b) = \begin{cases} +1 & \text{se } w^T x + b \ge 0 \\ -1 & \text{se } w^T x + b < 0 \end{cases}$$
* **Regola di Aggiornamento On-Line**:
  $$w \leftarrow w + \eta (y_i - o_i) x_i$$
* **Teorema di Convergenza di Novikoff (1962)**:
  Se il dataset è linearmente separabile con margine $\gamma > 0$, l'algoritmo convergerà commettendo al massimo $k$ errori:
  $$k \le \frac{R^2}{\gamma^2} \quad \text{con } R = \max_{i} \|x_i\|_2$$
* **Dimostrazione del Bound di Novikoff**:
  1. Ricorrenza sul limite superiore $\|w_k\|^2 \le k \eta^2 R^2$.
  2. Ricorrenza sul limite inferiore $w_k^T w^* \ge k \eta \gamma$.
  3. Applicando Cauchy-Schwarz: $(k \eta \gamma)^2 \le (w_k^T w^*)^2 \le \|w_k\|^2 \|w^*\|^2 \le k \eta^2 R^2 \implies k \le \frac{R^2}{\gamma^2}$.
* **Esempio Porta NOT**: $w = -2, b = +1 \implies f(x) = \text{sign}(-2x + 1)$.

---

#### B) LMS (Least Mean Squares) / Regola del Delta (Widrow-Hoff)
* **Errore Continuo su Input Netto**: $net_i = w^T x_i + b$.
* **Regola del Delta**:
  $$\Delta w = \eta (y_i - net_i) x_i \implies w^{(t+1)} = w^{(t)} + \eta (y_i - net_i) x_i$$

---

#### C) Soluzione analitica: Equazioni Normali (Regressione Lineare)
* **Loss Matriciale**: $L(w) = \frac{1}{N} \|X w - y\|_2^2 = \frac{1}{N} (X w - y)^T (X w - y)$.
* **Annullamento del Gradiente**:
  $$\nabla_w L(w) = \frac{2}{N} (X^T X w - X^T y) = 0 \implies X^T X w = X^T y$$
* **Soluzione in Forma Chiusa**:
  $$w^* = (X^T X)^{-1} X^T y = X^+ y$$
  dove $X^+$ è la Pseudoinversa di Moore-Penrose.

---

#### D) Linear Basis Expansion (LBE) & Teorema di Cover (1965)
* **Proiezione LBE**: $h(x) = w^T \phi(x) + b = \sum_{k=1}^K w_k \phi_k(x) + b$ con $\phi: \mathbb{R}^D \to \mathbb{R}^K$ ($K \gg D$).

---

### 1.3 k-Nearest Neighbors (k-NN) & Curse of Dimensionality (`ML-25-knn-v0.1`)
* **Regressione k-NN**: $\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$
* **Classificazione k-NN**: $\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$

---

## 📌 MODULO 2: Reti Neurali Multi-Layer & Backprop (Slides `NN-part1`, `NN-part2`)

### 2.1 Funzioni di Attivazione e Derivate
* **Sigmoide**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, derivata $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
* **Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, derivata $\tanh'(z) = 1 - \tanh^2(z)$.
* **ReLU**: $f(x) = \max(0, x)$, derivata $f'(x) = \mathbb{I}(x > 0)$.
* **LeakyReLU**: $f(x) = \max(\alpha x, x)$.
* **ELU**: $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$, derivata $f'(x) = \begin{cases} 1 & x > 0 \\ f(x) + \alpha & x \le 0 \end{cases}$.
* **Dimostrazione della necessità di attivazioni non lineari**: $o = W_L W_{L-1} \dots W_1 x = W_{tot} x$.

---

### 2.2 Teorema di Approssimazione Universale (Hornik 1989)
$$f(x) = \sum_{j=1}^M v_j \sigma(w_j^T x + b_j)$$

---

### 2.3 L'Algoritmo di Backpropagation (Derivazione Completa)
* **Forward Pass**: $net_j = \sum_u w_{ju} o_u$, $o_j = f_j(net_j)$.
* **Derivata del Peso**: $\frac{\partial E_p}{\partial w_{tu}} = -\delta_t o_u \implies \Delta w_{tu} = \eta \delta_t o_u$.
* **Delta Output $\delta_k$**: $\delta_k = (d_k - o_k) f'_k(net_k)$.
* **Delta Hidden $\delta_j$**: $\delta_j = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$.
* **Complessità**: $O(|W|)$ per epoca.
* **Vanishing Gradient Chain Rule**:
  $$\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$$

---

### 2.4 Formula Integrale dell'Aggiornamento Pesi
$$\Delta w_{tu}(t) = -\eta \frac{\partial E}{\partial w_{tu}(t)} + \alpha \Delta w_{tu}(t-1) - \eta \lambda w_{tu}(t)$$

---

## 📌 MODULO 3: Validazione & Model Selection (Slides `Valid1`, `Valid2`, `Valid3`)

### 3.1 K-Fold Cross-Validation
$$\bar{E}_{CV} = \frac{1}{K} \sum_{k=1}^K E_k, \quad \sigma_{CV} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (E_k - \bar{E}_{CV})^2}$$

---

## 📌 MODULO 4: Statistical Learning Theory, SVM & Bias-Varianza (Slides `SLT1`, `SVM`, `Bias-Variance`)

### 4.1 Decomposizione Bias-Varianza-Rumore (Dimostrazione Algebrica)
$$\mathbb{E}_{D,\epsilon}[(y - h_D(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D [ (h_D(x) - \bar{h}(x))^2 ]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$

---

### 4.2 Statistical Learning Theory (SLT) & Teorema di Radon
* **Bound di Vapnik**:
  $$R(h) \le R_{emp}(h) + \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$
* **Teorema di Radon & $VC = D + 1$**:
  Poiché per ogni insieme di $D+2$ punti in $\mathbb{R}^D$ si ha $\text{Conv}(A) \cap \text{Conv}(B) \neq \emptyset$, nessun iperpiano lineare può shatterare $D+2$ punti $\implies VC = D + 1$.

---

### 4.3 Support Vector Machines (SVM & SVR)

#### A) Massimo Margine Geometrico
$$M = \frac{2}{\|w\|}$$

#### B) Formulario Primale (Soft Margin)
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{sotto vincoli } y_i(w^T x_i + b) \ge 1 - \xi_i, \,\, \xi_i \ge 0$$

#### C) Formulazione Duale & Condizioni KKT
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{sotto } 0 \le \alpha_i \le C, \,\, \sum_{i=1}^N \alpha_i y_i = 0$$
* **Sparsità KKT**: $\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$.
* **Kernel RBF Gaussiano**: $K(x, z) = \exp(-\gamma \|x-z\|^2)$ dove $\gamma = \frac{1}{2\sigma^2}$.
* **Support Vector Regression (SVR)**: Loss $\epsilon$-insensitive $L_\epsilon(y, f(x)) = \max(0, |y - f(x)| - \epsilon)$ con slack doppie $(\xi_i, \xi_i^*)$.

---

## 📌 MODULO 5: Architetture Avanzate, Un-Supervised & Deep Learning (Slides `CNN`, `Deep`, `Rand`, `Unsupervised-SOM`, `RNN`, `SDL-Intro`)

### 5.1 Convolutional Neural Networks (CNN)
* **Convoluzione Discreta 2D**: $(I * K)[i, j] = \sum_m \sum_n I[i-m, j-n] K[m, n]$
* **Cross-Correlazione 2D (PyTorch)**: $(I \star K)[i, j] = \sum_m \sum_n I[i+m, j+n] K[m, n]$
* **`im2col`**: Trasformazione matriciale per GPU.

---

### 5.2 Deep Learning & Risultati di Profondità
* **No-Flattening**: Parità a $N$-bit risolta con $O(N)$ porte in $\log N$ strati vs $O(2^N)$ in 2 strati.

---

### 5.3 Algoritmi Costruttivi & Random NN
* **Cascade Correlation (Fahlman & Lebiere 1990)**:
  * Covarianza $S = \sum_{k \in \text{Output}} \left| \sum_p (o_p - \bar{o})(E_{p,k} - \bar{E}_k) \right|$
  * Ascesa del gradiente col segno PLUS: $\Delta w_j = +\eta \frac{\partial S}{\partial w_j}$
  * Pesi congelati (*frozen weights*).
* **Echo State Networks (ESN)**: Stato $x(t+1) = \tanh(W_{in} u(t+1) + W_{res} x(t))$, Readout $y(t+1) = W_{out} x(t+1)$.

---

### 5.4 Apprendimento Non Supervisionato: K-Means & SOM
* **Quantizzazione Vettoriale & Voronoi**: Celle $V_i = \{x : \|x - \mu_i\| \le \|x - \mu_j\|\}$. Distorsione globale $E = \sum_i \sum_{x \in V_i} \|x - \mu_i\|^2$.
* **SOM di Kohonen (1982)**: $\text{BMU} = \arg\min_i \|x - w_i\|$, update $w_i(t+1) = w_i(t) + \eta(t) h_{ic}(t) (x(t) - w_i(t))$ con vicinato gaussiano $h_{ic}(t) = \exp\left(-\frac{\|r_i - r_c\|^2}{2\sigma(t)^2}\right)$.
* **Autoencoders**: $\|x - \hat{x}\|^2 = \|x - f(g(x))\|^2$.

---

### 5.5 Recurrent Neural Networks (RNN)
* **Transizione di Stato**: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
* **Output**: $y_t = \text{softmax}(W_{hy} h_t + b_y)$
* **Gradient Clipping**: $g \leftarrow v \frac{g}{\|g\|_2}$

---

### 5.6 Domini Strutturati & Graph Neural Networks (GNN)
* **Message Passing**: $m_v^{(k)} = \text{AGGREGATE}\left(\{h_u^{(k-1)} : u \in N(v)\}\right)$, $h_v^{(k)} = \text{UPDATE}\left(h_v^{(k-1)}, m_v^{(k)}\right)$
* **NN4G**: $h_v^{(k)} = f\left( W_1 x_v + \sum W_{2,l} \sum_{u \in N(v)} h_u^{(l)} \right)$
* **Readout Globale**: $y_G = W_{out} \left( \sum_{v \in V} h_v \right)$
* **Word Embeddings**: $\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$
