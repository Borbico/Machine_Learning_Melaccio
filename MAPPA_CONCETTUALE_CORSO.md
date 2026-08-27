# Mappa Concettuale e Compendio delle Formule: Machine Learning
**Corso del Prof. Alessio Micheli — Università di Pisa**
*(Guida Integrale Arricchita: Teoria, Formule, Spiegazioni Dettagliate, Codice di Progetto, Dimostrazioni d'Esame e Formulario Completo di Damiano)*

---

## 🗺️ Mappa Concettuale Globale

```mermaid
flowchart TD
    ML["MACHINE LEARNING"] --> P1["1. IL FRAMEWORK FONDAMENTALE\n(I 4 Elementi di ogni algoritmo)"]
    ML --> P2["2. APPRENDIMENTO SUPERVISIONATO\n(Modelli Lineari, NNs, SVM, k-NN)"]
    ML --> P3["3. VALIDAZIONE & STATISTICAL LEARNING THEORY\n(Bias-Varianza, SLT, VC-Dim, SRM)"]
    ML --> P4["4. ARCHITETTURE AVANZATE & UN-SUPERVISED\n(Autoencoder, RNN, SOM, GNN, Random NN)"]
    ML --> P5["5. GUIDA INTEGRALE AL CODICE DEL PROGETTO\n(File .py e Notebook .ipynb)"]
    ML --> P6["6. DOMANDE TRABOCCHETTO DELL'ORALE\n(Domande 'Spicy' di Micheli)"]
    ML --> P7["7. COMPENDIO ESAME SCRITTO & ORALE\n(8 Dimostrazioni Matematiche Complete)"]

    P1 --> F1["Dati X, y (Train, Val, Test)"]
    P1 --> F2["Spazio Ipotesi H & Inductive Bias"]
    P1 --> F3["Loss Function L (MSE, MEE, BCE)"]
    P1 --> F4["Ottimizzatore (Gradient Descent, SGD)"]

    P2 --> M1["Modelli Lineari (Perceptron, LMS / Delta Rule)"]
    P2 --> M2["Reti Neurali Multi-layer (MLP, Activations, Backprop)"]
    P2 --> M3["Support Vector Machines (SVM, SVR, Kernel RBF)"]
    P2 --> M4["Instance-Based (k-Nearest Neighbors)"]

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

    P6 --> T1["8 Domande Trabocchetto da Lode con Risposta Modello"]
    P7 --> D1["Dimostrazioni Passo-Passo (Backprop, Bias-Var, Perceptron, SVM, Radon)"]
```

---

# 📌 SEZIONE 1: Il Framework Fondamentale & Modelli Lineari

### 1. Il Framework dei 4 Elementi di Ogni Algoritmo di ML
Ogni modello di Machine Learning trattato nel corso può essere formalmente analizzato e scomposto attraverso **4 componenti fondamentali**:

1. **I Dati ($D$) ed i Rischi (Generale vs Empirico)**:
   * Insieme delle coppie osservate $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$, dove $x_i \in \mathbb{R}^D$ rappresenta il vettore delle feature d'ingresso e $y_i \in \mathcal{Y}$ rappresenta il target.
   * Il dataset viene suddiviso in tre insiemi disgiunti: **Training Set** (addestramento pesi), **Validation Set** (Model Selection) e **Test Set** (Model Assessment).
   * **Rischio Generale / Errore Atteso ($R(h)$)**: L'errore medio teorico del modello sull'intera distribuzione congiunta $P(x, y)$ di tutti i possibili dati futuri (non calcolabile direttamente):
     $$R(h) = \int_{\mathcal{X} \times \mathcal{Y}} L(h(x), y) \, dP(x, y)$$
   * **Rischio Empirico ($R_{emp}(h)$)**: L'errore medio calcolato concretamente sul dataset osservato $D$ di $N$ campioni:
     $$R_{emp}(h) = \frac{1}{N} \sum_{i=1}^N L(h(x_i), y_i)$$

2. **Lo Spazio delle Ipotesi ($\mathcal{H}$)**:
   * L'insieme di tutte le funzioni $h: \mathcal{X} \to \mathcal{Y}$ che il modello scelto è teoricamente in grado di rappresentare al variare dei suoi parametri $w$.
   * **Inductive Bias (Bias Induttivo)**: L'insieme di assunzioni a priori sulla forma della funzione che il modello impone per poter generalizzare su dati non visti. Ad esempio, una regressione lineare assume che la relazione sia un iperpiano, mentre un albero di decisione assume confini a scalino paralleli agli assi.

3. **La Loss Function ($\mathcal{L}$)**:
   * La misura matematica dell'errore commesso dalla funzione di ipotesi $h(x)$ rispetto al target reale $y$.
   * **MSE Loss (Mean Squared Error)**: Usata principalmente nella regressione scalare.
     $$L_{MSE}(w) = \frac{1}{N} \sum_{i=1}^N (y_i - h(x_i))^2$$
     *Spiegazione*: Elevare al quadrato i residui garantisce che gli errori di segno opposto non si annullino a vicenda e penalizza maggiormente i grandi scarti (outliers).
   * **MEE Loss (Mean Euclidean Error — Target Multivariato ML CUP)**: Usata nella regressione vettoriale $K$-dimensionale (con $K=4$ per la CUP).
     $$L_{MEE}(w) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2 = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^K (y_{i,m} - \hat{y}_{i,m})^2}$$
     *Spiegazione*: Misura la reale distanza metrica in linea retta (norma $L_2$) tra il punto reale e quello predetto nello spazio $\mathbb{R}^4$. A differenza dell'MSE, il MEE mantiene l'unità di misura fisica originale del problema ed impedisce che target con valori numerici elevati dominino il gradiente a scapito di target con valori più piccoli.
   * **BCE Loss (Binary Cross-Entropy)**: Usata per la classificazione binaria ($y_i \in \{0, 1\}$).
     $$L_{BCE}(w) = -\frac{1}{N} \sum_{i=1}^N \big[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \big]$$
     *Spiegazione*: Deriva dalla massima verosimiglianza bernoulliana. Penalizza in modo logaritmico asintotico le predizioni errate fornite con elevata confidenza (es. predire $\hat{y} \to 0$ quando il target reale è $y=1$).

4. **L'Algoritmo di Ottimizzazione**:
   * La procedura numerica utilizzata per navigare la superficie di errore dello spazio dei parametri e trovare il vettore di pesi $w^*$ che minimizza la loss.
   * **Gradient Descent (Discesa del Gradiente)**: Aggiorna i pesi nella direzione opposta a quella del gradiente della loss:
     $$w^{(t+1)} = w^{(t)} - \eta \nabla L(w^{(t)})$$
     dove $\eta > 0$ è il **learning rate** (tasso di apprendimento).

---

### 2. Modelli Lineari: Perceptron e LMS (Delta Rule)

#### A) Il Perceptron (Rosenblatt 1958)
* **Modello**: Unità a Soglia Lineare (LTU). L'uscita è determinata dall'applicazione della funzione a gradino alla combinazione lineare degli input:
  $$f(x) = \text{sign}(w^T x + b) = \begin{cases} +1 & \text{se } w^T x + b \ge 0 \\ -1 & \text{se } w^T x + b < 0 \end{cases}$$
* **Iperpiano di Decisione**: La superficie $w^T x + b = 0$ divide lo spazio delle feature in due semispazi. Il vettore pesi $w$ è **ortogonale** all'iperpiano ed indica la direzione del semispazio positivo (+1).
* **Regola di Aggiornamento On-Line (Hebbian-like)**: Ad ogni esempio $x_i$ su cui il modello commette un errore ($o_i \neq y_i$):
  $$w \leftarrow w + \eta (y_i - o_i) x_i$$
  *Spiegazione Geometrica*: Se $y_i = +1$ ma $o_i = -1$, la quantità $(y_i - o_i) = +2$, quindi sommiamo $2\eta x_i$ a $w$, ruotando il vettore dei pesi verso il punto $x_i$ per includerlo nel semispazio positivo.
* **Teorema di Convergenza (Novikoff 1962)**: Se un dataset è linearmente separabile con un margine $\gamma > 0$, l'algoritmo convergerà commettendo al massimo un numero finito di errori $k \le \frac{R^2}{\gamma^2}$, dove $R = \max_i \|x_i\|$.
* **Limiti del Perceptron**: Può apprendere **solo funzioni linearmente separabili**. Non è in grado di risolvere funzioni logiche non separabili come l'**XOR** (Minsky & Papert 1969).

#### 💡 Esempio d'Esame: Creare la porta NOT con 1 Perceptrone
* Input $x \in \{0, 1\}$, Target $y = 1 - x$ (se $x=0 \to y=1$; se $x=1 \to y=0$).
* Esprimendo le etichette in $\{-1, +1\}$: per $x=0 \to y=+1$; per $x=1 \to y=-1$.
* Scegliamo pesi $w = -2$, bias $b = +1$.
* Verificando l'output $f(x) = \text{sign}(-2x + 1)$:
  * Per $x = 0 \implies \text{sign}(+1) = +1$ (Corretto).
  * Per $x = 1 \implies \text{sign}(-1) = -1$ (Corretto).

---

#### B) LMS (Least Mean Squares) / Regola del Delta (Widrow-Hoff)
* **Differenza Fondamentale dal Perceptron**: L'errore per l'aggiornamento pesi viene calcolato **prima** di applicare la funzione di attivazione a gradino, basandosi sulla risposta lineare continua $net_i = w^T x_i + b$.
* **Regola del Delta**:
  $$\Delta w = \eta (y_i - net_i) x_i$$
* **Significato Teorico**: Esegue una vera discesa del gradiente lungo la superficie dell'errore quadratico medio (MSE). Trattandosi di una superficie parabolica quadrata, l'LMS ammette **un unico minimo globale** ed evita i blocchi causati dalle derivate nulle delle funzioni a gradino.

#### C) Soluzione in Forma Chiusa per la Regressione Lineare (Equazioni Normali)
Per un problema di regressione lineare con loss MSE, il vettore ottimo dei pesi $w^*$ può essere ricavato analiticamente in forma chiusa (senza iterazioni di gradiente) risolvendo le **Equazioni Normali**:
$$w^* = (X^T X)^{-1} X^T y = X^+ y$$
dove $X^+ = (X^T X)^{-1} X^T$ rappresenta la **Matrice Pseudoinversa di Moore-Penrose**.

#### D) Teorema di Cover sulla Separabilità (Cover 1965)
* **Enunciato**: Un problema di classificazione vettoriale non linearmente separabile in uno spazio di input a bassa dimensione $\mathbb{R}^D$ ha una probabilità significativamente più elevata di diventare **linearmente separabile** se proiettato tramite una trasformazione non lineare in uno spazio delle feature a dimensione elevata $\mathbb{R}^K$ ($K \gg D$).
* **Importanza**: È la giustificazione teorica alla base sia delle *Linear Basis Expansions (LBE)* sia del *Kernel Trick nelle SVM* e dei layer nascosti nelle *Reti Neurali*.

---

### 3. Funzioni di Attivazione e loro Proprietà

Le funzioni di attivazione introducono la **non-linearità** essenziale nelle reti neurali.

1. **Sigmoide (Logistica)**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{Derivata: } \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
   * *Proprietà*: Mappa qualunque valore reale nell'intervallo aperto $(0, 1)$, permettendo un'interpretazione probabilistica dell'uscita.
   * *Svantaggi*: Satura molto velocemente per $|z| > 4$, dove la derivata $\sigma'(z) \to 0$. Questo causa il fenomeno del **Vanishing Gradient** durante la Backpropagation su reti profonde.

2. **Tanh (Tangente Iperbolica)**:
   $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \text{Derivata: } \tanh'(z) = 1 - \tanh^2(z)$$
   * *Proprietà*: Mappa l'output nell'intervallo $(-1, 1)$. È **zero-centered** (centrata sullo zero), il che evita sbilanciamenti sistematici nei gradienti durante l'addestramento ed accelera la convergenza rispetto alla Sigmoide.

3. **ReLU (Rectified Linear Unit)**:
   $$f(x) = \max(0, x), \quad \text{Derivata: } f'(x) = \begin{cases} 1 & \text{se } x > 0 \\ 0 & \text{se } x \le 0 \end{cases}$$
   * *Significato della derivata (Domanda Frequente d'Esame)*: Agisce come un **interruttore / maschera binaria (0/1)** basato sul valore di $net$.
   * *Perché risolve il Vanishing Gradient?* Per tutti gli input positivi ($x > 0$), la derivata è **esattamente 1**. Il gradiente viene trasmesso all'indietro moltiplicato per 1, senza alcuna attenuazione o degradamento attraverso molti strati.
   * *Svantaggio ("Dying ReLU")*: Se un neurone riceve un aggiornamento pesi che lo porta ad avere sempre $net < 0$, la sua derivata sarà costantemente 0 ed il neurone non si aggiornerà mai più. Soluzione: **LeakyReLU** $f(x) = \max(\alpha x, x)$ con $\alpha \approx 0.01$.

4. **ELU (Exponential Linear Unit)**:
   $$f(x) = \begin{cases} x & \text{se } x > 0 \\ \alpha (e^x - 1) & \text{se } x \le 0 \end{cases}, \quad \text{Derivata: } f'(x) = \begin{cases} 1 & \text{se } x > 0 \\ f(x) + \alpha & \text{se } x \le 0 \end{cases}$$
   * *Proprietà*: Mantiene i vantaggi della ReLU per $x > 0$, ma garantisce una transizione fluida e continua per valori negativi $x \le 0$, spingendo la media delle attivazioni verso lo zero ed eliminando totalmente il problema delle unità 'morte'.

#### ❓ Perché NON possiamo usare solo attivazioni lineari?
Se usassimo funzioni di attivazione lineari $f(z) = c \cdot z$ in tutti gli strati di una rete neurale a $L$ layer, l'uscita complessiva sarebbe:
$$o = W_L (W_{L-1} (\dots (W_1 x))) = (W_L \cdot W_{L-1} \dots W_1) x = W_{tot} x$$
La composizione di trasformazioni lineari è algebricamente equivalente ad un **singolo modello lineare**. La rete perderebbe tutta la capacità di approssimare confini di decisione non lineari o funzioni complesse.

---

# 📌 SEZIONE 2: Reti Neurali Multi-Layer (MLP) & Backpropagation

### 1. Architettura Multi-Layer Perceptron (MLP)
Un MLP è una rete feed-forward composta da un layer di input, uno o più **strati nascosti (hidden layers)** e uno strato di output.
* **Teorema di Approssimazione Universale (Hornik 1989)**: Una rete feed-forward con un singolo strato nascosto contenente un numero sufficiente di neuroni ed attivazioni continue non lineari (es. Sigmoide o Tanh) può approssimare qualsiasi funzione continua su un insieme compatto con un grado di precisione arbitrario $\epsilon > 0$.

---

### 2. Spiegazione Dettagliata degli Iperparametri di Addestramento

Nell'aggiornamento pesi della rete neurale:
$$\Delta w_{tu}(t) = -\eta \frac{\partial E}{\partial w_{tu}(t)} + \alpha \Delta w_{tu}(t-1) - \eta \lambda w_{tu}(t)$$

1. **Learning Rate ($\eta$)**:
   * Determina l'ampiezza del passo compiuto lungo la direzione del gradiente.
   * Se $\eta$ è troppo elevato: l'ottimizzatore salta oltre il minimo, generando oscillazioni instabili o divergenza.
   * Se $\eta$ è troppo piccolo: la convergenza richiede un numero enorme di epoche e la rete corre il rischio di bloccarsi in minimi locali o punti di sella non ottimali.

2. **Momentum ($\alpha \in [0, 1)$)**:
   * Aggiunge una quota dell'aggiornamento pesi effettuato all'epoca precedente $\Delta w(t-1)$.
   * *Spiegazione Fisica*: Simula l'inerzia di una massa che rotola sulla superficie di errore. Smorza le oscillazioni ad alta frequenza nelle direzioni ad alta curvatura (canyon) ed accelera la discesa nelle direzioni a pendenza constante.

3. **Weight Decay / Regolarizzazione $L_2$ ($\lambda$)**:
   * Aggiunge alla funzione di costo una penalizzazione proporzionale alla norma al quadrato dei pesi $\frac{\lambda}{2} \|w\|^2$.
   * *Spiegazione*: Costringe la discesa del gradiente a ridurre la magnitudo dei pesi ad ogni epoca ($w \leftarrow (1 - \eta\lambda)w - \eta \nabla E$). Pesi più piccoli producono superfici di risposta più levigate (*smooth*), limitando la capacità della rete ed impedendo l'overfitting.

---

### 3. La Backpropagation (Algoritmo di Retropropagazione dell'Errore)

La Backpropagation calcola in modo efficiente il gradiente dell'errore rispetto a tutti i pesi della rete applicando la **Chain Rule**.

#### Il Flusso dell'Algoritmo:
1. **Forward Pass (Da sinistra a destra)**:
   Per ciascun neurone $j$ dello strato nascosto e $k$ dello strato di output, si calcola l'ingresso netto e l'uscita:
   $$net_j = \sum_u w_{ju} o_u, \quad o_j = f_j(net_j)$$
2. **Backward Pass (Da destra a sinistra)**:
   Si calcola il fattore di errore locale $\delta$ partendo dallo strato di output e retropropagandolo verso gli strati nascosti:
   * **Neurone di Output $k$**:
     $$\delta_k = (d_k - o_k) f'_k(net_k)$$
   * **Neurone Nascosto $j$**:
     $$\delta_j = \left( \sum_{k \in \text{Output}} \delta_k w_{kj} \right) f'_j(net_j)$$
3. **Aggiornamento Pesi**:
   $$\Delta w_{tu} = \eta \delta_t o_u$$

#### ⚡ Perché la Backpropagation è Efficiente?
Calcolare le derivate parziali in modo ingenuo per ciascun peso richiederebbe un tempo quadratico $O(|W|^2)$. La Backpropagation riutilizza i valori dei $\delta_k$ del layer successivo per calcolare i $\delta_j$ del layer precedente, fattorizzando i calcoli intermedi e riducendo la complessità a **$O(|W|)$ (lineare nel numero di pesi)**.

#### 📉 Analisi Matematica del Vanishing Gradient tramite Chain Rule
Considerando la derivata rispetto al peso $w_1$ del primo strato in una rete profonda a $L$ layer:
$$\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$$
Poiché per attivazioni sigmoidali la derivata $f'_l(net_l) \le 0.25$, la produttoria di $L-1$ fattori inferiori a 1 tende esponenzialmente a zero per $L \ge 4$, azzerando gli aggiornamenti dei pesi iniziali.

---

# 📌 SEZIONE 3: Support Vector Machines (SVM & SVR)

### 1. Il Principio del Massimo Margine
Nelle Support Vector Machines per la classificazione binaria, l'obiettivo non è semplicemente trovare un iperpiano separatore qualunque, ma trovare l'iperpiano che **massimizza il margine geometrico $M$**.
* **Margine Geometrico ($M = \frac{2}{\|w\|}$)**: La distanza tra i due iperpiani paralleli di bordo che delimitano le due classi.
* **Perché massimizzare il margine evita l'overfitting?**
  Massimizzare $M$ equivale a minimizzare la norma $\|w\|^2$. In base alla Statistical Learning Theory, la VC-dimension degli iperpiani con margine ampio è rigorosamente limitata $VC \le \frac{R^2}{\gamma^2} + 1$, rendendo il modello indipendente dalla dimensionalità dello spazio di input $D$.

---

### 2. Soft Margin SVM ($C$ e Slack Variables $\xi_i$)
Per gestire dataset non perfettamente separabili o rumorosi, si introducono le **variabili slack $\xi_i \ge 0$**:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{sotto vincoli } y_i(w^T x_i + b) \ge 1 - \xi_i, \,\, \xi_i \ge 0$$

#### Significato Geometrico delle Slack Variables $\xi_i$:
* $\xi_i = 0$: Il punto si trova correttamente all'esterno della fascia del margine.
* $0 < \xi_i \le 1$: Il punto si trova all'interno del margine, ma è classificato correttamente.
* $\xi_i > 1$: Il punto viola il margine e cade dal lato sbagliato dell'iperpiano (errore di classificazione).

#### Il Ruolo del Parametro di Regolarizzazione $C$:
* **$C \to \infty$**: Penalizzazione infinita per gli errori. Si forza un margine rigido (*Hard Margin*). Se i dati non sono separabili, l'ottimizzatore non converge. Se separabili, crea un margine strettissimo legato agli outlier (overfitting).
* **$C$ piccolo**: Tolleranza elevata per le violazioni. Il modello dà priorità all'ampiezza del margine piuttosto che al corretto posizionamento dei singoli punti di train (regolarizzazione elevata, rischia underfitting se troppo piccolo).

---

### 3. Formulazione Duale e Sparsità KKT
Tramite i moltiplicatori di Lagrange $\alpha_i \ge 0$, si ottiene la **Formulazione Duale**:
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{sotto } 0 \le \alpha_i \le C, \,\, \sum_{i=1}^N \alpha_i y_i = 0$$

#### Sparsità KKT e Vettori di Supporto:
Dalle condizioni di complementarietà KKT ($\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$), deriva una proprietà fondamentale:
* Per tutti i punti ben classificati all'esterno del margine, **$\alpha_i = 0$**.
* Solo per i punti che giacciono sul bordo del margine o lo violano si ha **$\alpha_i > 0$**. Questi punti sono i **Vettori di Supporto (Support Vectors)**.
* La funzione di decisione finale dipende **esclusivamente dai Support Vector**:
  $$f(x) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right)$$

---

### 4. Kernel Trick e Kernel RBF Gaussiano

#### Il Kernel Trick:
Consente di proiettare implicitamente i dati d'ingresso da uno spazio $\mathbb{R}^D$ ad uno spazio di feature a dimensionalità superiore $\Phi(x) \in \mathcal{F}$ dove i dati diventano linearmente separabili, **senza mai calcolare esplicitamente le coordinate $\Phi(x)$**, ma calcolando solo il prodotto scalare $K(x_i, x_j) = \Phi(x_i)^T \Phi(x_j)$.

#### Kernel RBF Gaussiano (Radial Basis Function):
$$K(x, z) = \exp(-\gamma \|x - z\|^2)$$
* **Significato del Parametro $\gamma$**:
  * $\gamma = \frac{1}{2\sigma^2}$ determina il raggio di influenza di ciascun vettore di supporto.
  * **$\gamma$ elevato**: Raggio di influenza molto piccolo. La funzione di decisione crea "bolle" attorno ai singoli Support Vector (confine di decisione frastagliato $\to$ **Overfitting**).
  * **$\gamma$ basso**: Raggio di influenza ampio. La funzione di decisione è liscia e quasi lineare (confine di decisione rigido $\to$ **Underfitting**).

---

### 5. Support Vector Regression (SVR ed la $\epsilon$-Tube)
Nella SVR l'obiettivo è trovare una funzione $f(x) = w^T x + b$ che devii dai target reali $y_i$ di al massimo una quantità $\epsilon$.
* **$\epsilon$-Insensitive Loss**:
  $$L_\epsilon(y, f(x)) = \max(0, |y - f(x)| - \epsilon)$$
  *Spiegazione*: Costruisce un "tubo" di tolleranza di raggio $\pm \epsilon$ attorno alla curva predetta. Qualsiasi errore compreso all'interno del tubo viene considerato nullo ($\text{loss} = 0$). Solo i punti al di fuori del tubo generano un costo ed operano come Support Vector della regressione.

---

# 📌 SEZIONE 4: Statistical Learning Theory (SLT) & Validazione

### 1. La Decomposizione Bias-Varianza-Rumore
Per una funzione obiettivo $y = f(x) + \epsilon$ con rumore a media nulla e varianza $\sigma^2$, l'Errore Quadratico Medio atteso su un nuovo punto $x$ si scompone rigorosamente in 3 componenti:

$$\mathbb{E}[(y - h_D(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}(x)^2} + \underbrace{\mathbb{E}_D [ (h_D(x) - \bar{h}(x))^2 ]}_{\text{Varianza}(x)} + \underbrace{\sigma^2}_{\text{Rumore Irriducibile}}$$

1. **Bias$^2$ (Errore di Approssimazione)**:
   * Misura lo scarto tra la funzione reale $f(x)$ e la media delle predizioni $\bar{h}(x)$ del nostro modello al variare dei dataset di addestramento. Un Bias alto indica un modello troppo semplice o uno spazio di ipotesi incapace di rappresentare la complessità del dato (**Underfitting**).
2. **Varianza (Errore di Stima)**:
   * Misura quanto le singole predizioni $h_D(x)$ oscillano attorno alla loro media $\bar{h}(x)$ a seconda dello specifico dataset $D$ estratto. Una Varianza alta indica che il modello è estremamente sensibile al rumore del training set (**Overfitting**).
3. **Rumore Irriducibile ($\sigma^2$)**:
   * L'incertezza stocastica intrinseca dei dati stessi (dovuta a variabili non misurate o errori degli strumenti). Non può essere eliminata da alcun modello.

---

### 2. VC-Dimension (Vapnik-Chervonenkis)

* **Shattering (Frammentazione)**: Un insieme di $N$ punti è shatterato dallo spazio delle ipotesi $\mathcal{H}$ se per qualunque delle $2^N$ possibili etichettature binaria (+1 / -1), esiste almeno un'ipotesi $h \in \mathcal{H}$ in grado di classificarli tutti correttamente.
* **Definizione di VC-Dimension ($h_{VC}$)**: La **massima cardinalità $N$** di punti per cui esiste ALMENO UN insieme di $N$ punti shatterabile da $\mathcal{H}$. Se $\mathcal{H}$ può shatterare insiemi arbitrariamente grandi, allora $h_{VC} = \infty$.

#### Valori della VC-Dimension per i Modelli del Corso:
* **Iperpiani lineari in $\mathbb{R}^D$ (Perceptron)**: $h_{VC} = D + 1$.
* **Support Vector Machines**: $h_{VC} \le \min\left(D, \frac{R^2}{\gamma^2}\right) + 1$.
* **Reti Neurali Multi-Layer**: Cresce col numero di pesi $O(|W| \log |W|)$.
* **1-Nearest Neighbor (1-NN)**: $h_{VC} = \infty$ (memorizza qualunque insieme di punti creando poligoni di Voronoi).

---

### 3. Structural Risk Minimization (SRM) & Bound di Vapnik
Il Rischio Reale $R(h)$ (errore su dati futuri) è limitato superiormente con probabilità $1-\eta$ dalla somma dell'Errore Empirico sul train $R_{emp}(h)$ e del termine di complessità VC:

$$R(h) \le R_{emp}(h) + \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$

* **Principio SRM**: Organizza le ipotesi in una successione di sottospazi annidati $\mathcal{H}_1 \subset \mathcal{H}_2 \dots \subset \mathcal{H}_k$ con VC-dimension crescente. SRM seleziona l'ipotesi che minimizza il bound complessivo (bilanciando errore empirico e capacità).
* **Early Stopping come SRM**: Interrompere l'addestramento ad un numero finito di epoche limita la magnitudo dei pesi e la dimensione VC effettiva della rete neurale, prevenendo l'overfitting.

---

### 4. Schemi di Validazione e Prevenzione del Data Leakage

#### A) K-Fold Cross-Validation
Sulle dimensioni ridotte del dataset (es. 400 campioni del dev set CUP), l'Hold-Out semplice ha un'elevata varianza. Si suddivide il dataset in $K$ partizioni (fold). A turno, 1 fold funge da Validation ed i restanti $K-1$ da Training. La metrica finale è la media sui $K$ cicli:
$$\bar{E}_{CV} = \frac{1}{K} \sum_{k=1}^K E_k, \quad \sigma_{CV} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (E_k - \bar{E}_{CV})^2}$$

#### B) Prevenzione del Data Leakage (Contaminazione dei Dati)
* **Regola Tassativa**: Qualunque trasformazione sui dati (es. `StandardScaler`, `RobustScaler`, selezione delle feature) **deve apprendere i suoi parametri (media, deviazione standard, mediana) ESCLUSIVAMENTE dal training fold di ciascun ciclo di K-Fold**.
* **Errore Grave**: Se si applicasse lo scaler sull'intero dataset prima del K-Fold, lo scaler vedrebbe le medie del Validation set in anticipo. Questo costituisce **Data Leakage**, producendo stime di errore ottimistiche e non veritiere.

#### C) Nested Cross-Validation (Doppio Ciclo)
* Utilizzata per valutare la reale capacità di generalizzazione quando si esegue la ricerca degli iperparametri (GridSearch/Optuna).
* **Inner Loop (K-Fold Interno)**: Usato esclusivamente per la **Model Selection** (trovare la configurazione migliore di iperparametri $h^*$).
* **Outer Loop (K-Fold Esterno)**: Usato per il **Model Assessment** (valutare le prestazioni del modello $h^*$ su fold esterni che non hanno mai partecipato alla ricerca degli iperparametri, eliminando il *selection bias*).

---

### 5. Formulazione Formale di k-NN ed Ensemble Learning

#### A) k-Nearest Neighbors (k-NN)
Definito $N_K(x)$ l'insieme dei $K$ campioni del training set più vicini all'input $x$ secondo una metrica di distanza (es. Euclidea):
* **Regressione k-NN (Media dei Vicini)**:
  $$\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$
* **Classificazione k-NN (Voto a Maggioranza)**:
  $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$$

#### B) Ensemble Learning — Schema Voto del Comitato
Dato un comitato di $M$ modelli indipendenti $h_1, h_2, \dots, h_M$:
* **Regressione (Media del Comitato)**:
  $$h_{ens}(x) = \frac{1}{M} \sum_{m=1}^M h_m(x)$$
* **Classificazione Binaria / Multiclasse (Voto di Maggioranza del Comitato)**:
  $$h_{ens}(x) = \arg\max_{c} \sum_{m=1}^M \mathbb{I}(h_m(x) = c)$$

---

# 📌 SEZIONE 5: Architetture Avanzate

### 1. Cascade Correlation (Algoritmo Costruttivo)
Sviluppato da Fahlman & Lebiere (1990), è un algoritmo costruttivo per reti neurali:
1. Parte da una rete minima senza neuroni nascosti (solo connessioni dirette input-output).
2. Addestra la rete fino a stagnazione dell'errore.
3. Se l'errore è insoddisfacente, genera una popolazione di **neuroni candidati** che ricevono connessioni dagli input e da tutti i neuroni nascosti aggiunti in precedenza.
4. Addestra i candidati per **massimizzare la covarianza $S$** con l'errore residuo della rete:
   $$S = \sum_{k \in \text{Output}} \left| \sum_{p} (o_p - \bar{o})(E_{p,k} - \bar{E}_k) \right|$$
5. Il candidato con la covarianza $S$ più alta viene inserito nella rete e **i suoi pesi d'ingresso vengono congelati per sempre (*frozen weights*)**.
6. Si ri-addestrano solo i pesi dello strato di output e si ripete il ciclo.
* **Vantaggi**: Non richiede di specificare a priori il numero di neuroni nascosti, apprende molto velocemente ed evita il *moving target problem*.

---

### 2. Autoencoders (Undercomplete vs Overcomplete)
Reti neurali per l'apprendimento non supervisionato composte da un **Encoder** $z = g(x)$ ed un **Decoder** $\hat{x} = f(z)$, addestrate per minimizzare l'errore di ricostruzione $\|x - \hat{x}\|^2$.
* **Undercomplete Autoencoder ($dim(z) < dim(x)$)**: Il bottleneck ristretto costringe la rete a comprimere l'informazione, estraendo una rappresentazione latente non lineare simile ad una PCA non lineare.
* **Overcomplete Autoencoder ($dim(z) > dim(x)$)**: Lo strato nascosto ha una dimensione maggiore dell'input. Per evitare che la rete impari una banale funzione identità, si aggiungono vincoli di regolarizzazione come la **Sparsity Penalty** o il rumore d'ingresso (**Denoising Autoencoders**).

---

### 3. Random Neural Networks & Reservoir Computing (Echo State Networks)
* **Principio**: I pesi degli strati nascosti (o del bacino ricorsivo *reservoir*) vengono **inizializzati casualmente e congelati permanently**.
* **Addestramento**: Si addestrano solo i pesi del layer di output tramite una semplice regressione lineare chiusa (LMS o Ridge Regression).
* **Vantaggi**: Velocità di addestramento istantanea, assenza totale di problemi di minima locali e convergenza garantita.

---

### 4. Self-Organizing Maps (SOM di Kohonen 1982)
* **Apprendimento Topologico Non Supervisionato**: Mappa campioni ad alta dimensione su una griglia discreta bidimensionale (es. $10 \times 10$), preservando la vicinanza topologica dei dati.
* **Algoritmo**:
  1. Per ogni input $x$, individua il neurone della griglia i cui pesi sono più vicini all'input (**Best Matching Unit - BMU**).
  2. Aggiorna i pesi del BMU e dei suoi neuroni vicini sulla griglia:
     $$w_i(t+1) = w_i(t) + \eta(t) h_{ic}(t) \big( x(t) - w_i(t) \big)$$
     dove $h_{ic}(t) = \exp\left(-\frac{\|r_i - r_c\|^2}{2\sigma(t)^2}\right)$ è una funzione di vicinato gaussiana decrescente col tempo.

---

### 5. Graph Neural Networks (Message Passing, NN4G & Readout)
Progettate per operare su dati strutturati a grafo $G = (V, E)$ non euclidei.
* **Message Passing Generico**:
  $$m_v^{(k)} = \text{AGGREGATE}^{(k)} \left( \left\{ h_u^{(k-1)} : u \in N(v) \right\} \right), \quad h_v^{(k)} = \text{UPDATE}^{(k)} \left( h_v^{(k-1)}, m_v^{(k)} \right)$$
* **Formulazione Ricorsiva NN4G (Neural Network for Graphs)**:
  $$h_v^{(k)} = f\left( W_1 x_v + \sum_{l=1}^{k-1} W_{2,l} \sum_{u \in N(v)} h_u^{(l)} \right)$$
* **Readout Globale sul Grafo (Prediction)**:
  $$y_G = \text{Readout}\left( \sum_{v \in V} h_v \right) = W_{out} \left( \sum_{v \in V} h_v \right)$$

---

### 6. Quantizzazione Vettoriale, K-Means & Celle di Voronoi (Apprendimento Non Supervisionato)
* **Cella di Voronoi ($V_i$)**: La regione dello spazio costituita da tutti i punti $x$ per cui il centroide $\mu_i$ è il più vicino:
  $$V_i = \{x \in \mathcal{X} : \|x - \mu_i\| \le \|x - \mu_j\| \,\, \forall j \neq i\}$$
* **Errore Globale di Quantizzazione / Distorsione**:
  * *Discreto*: $E = \sum_{i=1}^k \sum_{x \in V_i} \|x - \mu_i\|^2$
  * *Continuo*: $E = \sum_{i=1}^k \int_{V_i} \|x - \mu_i\|^2 p(x) \, dx$
* **Regola di Aggiornamento On-line (Discesa del Gradiente per K-means)**:
  $$\mu_i \leftarrow \mu_i + \eta (x - \mu_i)$$

---

### 7. Convolutional Neural Networks (CNN) & Operatori
* **Convoluzione Continua 1D**: $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau$
* **Convoluzione Discreta 2D**: $(I * K)[i, j] = \sum_m \sum_n I[i-m, j-n] K[m, n]$
* **Cross-Correlazione 2D (usata nei framework di Deep Learning come PyTorch)**:
  $$(I \star K)[i, j] = \sum_m \sum_n I[i+m, j+n] K[m, n]$$
  *Differenza*: Non ribalta spazialmente il kernel $K$, consentendo una parallelizzazione GPU estremamente efficiente tramite l'operatore `im2col` (Image-to-Column).

---

### 8. Recurrent Neural Networks (RNN)
Modelli per sequenze temporali dotati di uno stato nascosto di memoria $h_t$:
* **Transizione di Stato**: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
* **Output del Layer**: $y_t = \text{softmax}(W_{hy} h_t + b_y)$

---

### 9. Word Embeddings & Rappresentazioni Distribuite
Nel Deep Learning NLP, le rappresentazioni distribuite dense catturano relazioni geometrico-semantiche nello spazio vettoriale:
$$\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$$

---

### 6. Fenomeni Moderni: Double Descent & Lottery Ticket Hypothesis
* **Double Descent (Nakkiran et al. 2019)**: All'aumentare della complessità del modello o delle epoche, l'errore di test prima decresce, poi aumenta vicino al limite di interpolazione (overfitting classico), e poi **scende di nuovo nel regime fortemente sopra-parametrizzato**, sfidando il trade-off Bias-Varianza classico.
* **Lottery Ticket Hypothesis (Frankle & Carbin 2018)**: Una rete neurale densa e grande inizializzata casualmente contiene una sottorete isolata (*"biglietto vincente"*) che, se addestrata da sola fin dall'inizio con i suoi pesi iniziali originari, raggiunge accuratezza pari all'intera rete in meno epoche.

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
