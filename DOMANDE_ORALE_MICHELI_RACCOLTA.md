# 🎓 Raccolta Ufficiale Domande d'Orale del Prof. Micheli
**Corso di Machine Learning — Università di Pisa**  
*(Compendio Integrale Risolto delle Domande Reali Fatte agli Esami Orali con Formule, Dimostrazioni, Diagrammi e Spiegazioni da Lode)*

---

## 🗺️ Indice Macro-Argomenti

1. **[MODULO 1] Reti Neurali Multi-Layer (MLP), Attivazioni & Backprop**
2. **[MODULO 2] Support Vector Machines (SVM), SVR & Kernel Methods**
3. **[MODULO 3] Statistical Learning Theory (SLT), VC-Dimension & SRM**
4. **[MODULO 4] Modelli Lineari, Perceptrone & Novikoff**
5. **[MODULO 5] Deep Learning, Fenomeni Moderni & Architetture Avanzate (CNN, RNN, GNN, Autoencoders, ESN, SOM, Boosting, Dropout)**

---

# 📌 MODULO 1: Reti Neurali Multi-Layer (MLP), Attivazioni & Backprop

### Q1: *"Perché si usa la Sigmoide? Perché NON usiamo la Step Function (a gradino) per la Backpropagation? Qual è il termine matematico che il professore cerca?"*
* **Risposta Modello**:
  * **Il Termine Matematico**: La Step Function $\text{sign}(z)$ **NON È DERIVABILE** (presenta una discontinuità a salto in $z=0$ e ha derivata identicamente nulla quasi ovunque $\frac{d}{dz}\text{sign}(z) = 0$ per $z \neq 0$).
  * **Perché impedisce la Backpropagation**: La Backpropagation si basa sulla regola della catena ($\text{Chain Rule}$) per calcolare il gradiente della Loss rispetto ai pesi: $\frac{\partial E}{\partial w} = \frac{\partial E}{\partial net} \cdot \frac{\partial net}{\partial w}$. Se la derivata dell'attivazione $f'(net)$ è $0$ quasi ovunque, i segnali d'errore locali $\delta$ si annullano e l'aggiornamento pesi $\Delta w = \eta \delta o$ si blocca completamente.
  * **Perché la Sigmoide**: La Sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$ è una funzione **continua e infinitamente derivabile ($C^\infty$)** con derivata elegante:
    $$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$
    Fornisce un segnale di gradiente continuo proporzionale alla risposta dell'unità.
* **Svantaggi della Sigmoide**:
  1. *Vanishing Gradient*: Per $|z| > 4$, $\sigma'(z) \to 0$ (saturazione), azzerando la produttoria dei gradienti nelle reti profonde.
  2. *Non Zero-Centered*: $\sigma(z) \in (0, 1)$, il che causa oscillazioni a zig-zag nell'aggiornamento dei pesi.

---

### Q2: *"Cos'è la ReLU? Scrivi la formula, la derivata, il grafico ed i vantaggi/svantaggi. Perché risolve il Vanishing Gradient?"*
* **Formula dell'Attivazione**:
  $$f(x) = \max(0, x) = \begin{cases} x & \text{se } x > 0 \\ 0 & \text{se } x \le 0 \end{cases}$$
* **Derivata Prima**:
  $$f'(x) = \begin{cases} 1 & \text{se } x > 0 \\ 0 & \text{se } x < 0 \end{cases}$$
* **Grafico ASCII dell'Attivazione e della Derivata**:
```
     Funzione ReLU f(x)                  Derivata f'(x)
         y |                                 y |
           |   / (pendenza 1)                  |------- (valore 1)
           |  /                                |
   --------+--/----> x                 --------+--------> x
   (0 per x<0)                         (0 per x<0)
```
* **Perché risolve il Vanishing Gradient per $x > 0$**:
  Nella produttoria della Chain Rule $\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) x$, per le sigmoidali $f'_l \le 0.25$, rendendo il prodotto tendente a zero. Per la ReLU, per tutti i neuroni attivi ($x > 0$), $f'(net) = 1$ in modo **costante**, trasmettendo il gradiente senza alcuna attenuazione.
* **Problema della "Dying ReLU" & Alternative**:
  Se un gradiente forte spinge $net < 0$, $f'(x)=0$ e il neurone "muore". Soluzioni: **LeakyReLU** $f(x) = \max(\alpha x, x)$ ($\alpha \approx 0.01$), **ELU**, **GELU**.

---

### Q3: *"Perché NON possiamo usare solo funzioni di attivazione LINEARI in una Rete Neurale?"*
* **Risposta Modello**:
  Se usassimo attivazioni lineari $f(z) = c \cdot z$ in tutti gli strati, l'uscita dell'ultimo layer $L$ sarebbe la composizione di trasformazioni lineari:
  $$o = W_L (W_{L-1} (\dots (W_1 x + b_1) \dots + b_{L-1})) + b_L = W_{tot} x + b_{tot}$$
  La rete neurale profundissima crollerebbe algebricamente in un **singolo modello lineare**, perdendo ogni capacità espressiva non lineare e la capacità di separare problemi complessi (come l'XOR).

---

### Q4: *"Qual è la differenza di espressività tra Reti Neurali Profonde (Deep NN) e Reti Superficiali (Shallow NN)?"*
* **Risposta Modello**:
  * **Reti Superficiali (Shallow NN — 1 Hidden Layer)**: Per il Teorema di Approssimazione Universale, un singolo strato nascosto può approssimare qualsiasi funzione continua. Tuttavia, per rappresentare funzioni complesse altamente oscillanti (come la parità a $D$-bit), una rete superficiale richiede un numero **esponenziale di neuroni nascosti $O(2^D)$**, rendendo l'addestramento impraticabile.
  * **Reti Profonde (Deep NN — $L$ Layer)**: Possono rappresentare la stessa funzione complessa con un numero **polinomiale di neuroni $O(D)$**, componendo astrazioni gerarchiche strato dopo strato. Ogni strato profondo ripartisce lo spazio delle feature in modo composizionale, aumentando in modo esponenziale il numero di regioni lineari trattabili a parità di parametri.

---

### Q5: *"Qual è la formula dell'output di una Rete Neurale con 1 Strato Nascosto (1 Hidden Layer NN)?"*
* **Formula Esplicita**:
  $$o_k = f_{out} \left( \sum_{j=1}^M w_{kj} \cdot f_{hid} \left( \sum_{i=1}^D w_{ji} x_i + b_j \right) + b_k \right)$$
  dove $D$ è il numero di input, $M$ è il numero di neuroni nascosti, $f_{hid}$ è l'attivazione nascosta (es. ReLU/Sigmoide), e $f_{out}$ è l'attivazione di output (es. Identità per regressione, Sigmoide per classificazione binaria).

---

### Q6: *"Enuncia precisamente il Teorema di Approssimazione Universale di MLP (Universal Approximation Theorem)."*
* **Enunciato (Cybenko 1989, Hornik 1989)**:
  Sia $\sigma(\cdot)$ una funzione di attivazione continua, non costante e limitata (es. Sigmoide). Per qualsiasi funzione continua $f(x)$ definita su un insieme compatto $K \subset \mathbb{R}^D$ e per ogni $\epsilon > 0$, esiste un numero finito di neuroni $M$ ed un insieme di pesi $\{v_j, w_j, b_j\}$ tali che la rete neurale a singolo strato nascosto:
  $$g(x) = \sum_{j=1}^M v_j \sigma(w_j^T x + b_j)$$
  soddisfa l'approssimazione uniforme:
  $$\|g(x) - f(x)\|_\infty < \epsilon \quad \forall x \in K$$

---

### Q7: *"Scrivi la formula dell'equazione del Momentum. Cos'è e perché si usa?"*
* **Formula Completa dell'Aggiornamento Pesi**:
  $$\Delta w_{tu}(t) = -\eta \frac{\partial E}{\partial w_{tu}(t)} + \alpha \Delta w_{tu}(t-1) - \eta \lambda w_{tu}(t)$$
  $$w^{(t+1)} = w^{(t)} + \Delta w(t)$$
* **Cos'è e Perché si Usa**:
  * Il Momentum $\alpha \in [0, 1)$ simula l'**inerzia fisica** di una sfera che rotola lungo la superficie dell'errore.
  * Aggiungendo una frazione dello spostamento precedente $\Delta w(t-1)$, smorza le oscillazioni ad alta frequenza nei direzioni a forte curvatura (canyon) ed accelera la convergenza lungo i settori pianeggianti.

---

### Q8: *"Cos'è il Dropout (Srivastava 2014) e perché previene l'overfitting?"*
* **Funzionamento**: Durante la fase di addestramento, per ogni mini-batch ciascun neurone dello strato nascosto viene "disattivato" (il suo output è impostato a zero) in modo stocastico ed indipendente con una probabilità $p$ (solitamente $p=0.5$).
* **Formula dell'Attivazione in Training vs Test**:
  $$\tilde{h}_j = \begin{cases} 0 & \text{con probabilità } p \\ \frac{h_j}{1-p} & \text{con probabilità } 1-p \end{cases} \quad (\text{In test: } \tilde{h}_j = h_j)$$
* **Perché funziona**:
  1. *Prevenzione del Co-adattamento (Co-adaptation)*: Impedisce che un neurone dipenda fortemente dalla presenza di uno specifico altro neurone, costringendo ciascun nodo ad apprendere feature utili ed autonome.
  2. *Ensemble Implicito*: Equivale ad addestrare ed effettuare la media di un comitato di $2^M$ sotto-reti neurali condivise.

---

### Q9: *"Perché l'Early Stopping può essere considerato una forma di regolarizzazione (SRM)?"*
* **Risposta Modello**:
  Durante l'addestramento per discesa del gradiente, i pesi partono da valori piccoli vicino allo zero ed aumentano la loro norma $\|w\|$ all'aumentare delle epoche.
  Limitare il numero di epoche interrompendo l'addestramento al minimo della loss di validazione (**Early Stopping**) pone un vincolo implicito alla crescita della norma $\|w\| \le B$.
  Poiché la VC-Dimension effettiva di una rete neurale cresce proporzionalmente alla norma dei suoi pesi, limitare le epoche limita direttamente la VC-Dimension e controlla il termine di confidenza VC di Vapnik, agendo esattamente come la regolarizzazione di Tikhonov $L_2$ (Weight Decay).

---

# 📌 MODULO 2: Support Vector Machines (SVM), SVR & Kernel Methods

### Q10: *"Qual è lo scopo delle SVM? Perché sono migliori di altri modelli lineari?"*
* **Scopo**: Trovare l'iperpiano separatore che **massimizza il margine geometrico di separazione** $M = \frac{2}{\|w\|}$ tra le classi.
* **Perché sono Migliori degli Altri Modelli Lineari**:
  1. *Unicità ed Ottimale*: Il Perceptrone o la Delta Rule trovano un iperpiano separatore qualsiasi tra i tanti possibili. La SVM trova l'**unico iperpiano ottimale a massimo margine**.
  2. *Minimizzazione della VC-Dimension*: Massimizzare il margine $M = 2/\|w\|$ equivale a minimizzare $\|w\|^2$, il che riduce direttamente la VC-Dimension dello spazio delle ipotesi ($VC \le \min(D, R^2/\gamma^2) + 1$), garantendo il miglior bound di generalizzazione della SLT.
  3. *Assenza di Minimi Locali*: Il problema della SVM è una Programmazione Quadratica Convessa con un **unico minimo globale**.

---

### Q11: *"La SVM risolve la Maledizione della Dimensionalità (Curse of Dimensionality)? Perché?"*
* **Risposta Modello**:
  **SÌ!** La SVM soffre molto meno la Curse of Dimensionality per due motivi fondamentali:
  1. *Indipendenza dalla Dimensione Fisica $D$*: La capacità espressiva e la VC-Dimension della SVM non dipendono dal numero teorico di feature fisiche $D$, ma dal margine geometrico $\gamma$: $VC \le \min\left(D, \frac{R^2}{\gamma^2}\right) + 1$. Se il margine $\gamma$ è ampio, la VC-dimension rimane piccolissima anche se $D \to \infty$.
  2. *Sparsità KKT & Kernel Trick*: La funzione di decisione finale dipende **esclusivamente dal prodotto scalare tra i dati ed i Vettori di Supporto** ($\alpha_i > 0$), ignorando tutti gli altri punti del dataset.

---

### Q12: *"Scrivi le equazioni della Forma Primale della Hard Margin SVM (Obiettivo, Vincoli, Lagrangiana, Condizioni di Ottimalità)."*
* **Problema di Ottimizzazione Primale**:
  $$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{sotto vincoli rigidi } y_i(w^T x_i + b) \ge 1 \quad \forall i=1\dots N$$
* **Funzione Lagrangiana**:
  $$L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^N \alpha_i \big[ y_i(w^T x_i + b) - 1 \big] \quad \text{con } \alpha_i \ge 0$$
* **Condizioni di Stazionarietà (Ottimalità di Primo Ordine)**:
  1. $\nabla_w L = 0 \implies w = \sum_{i=1}^N \alpha_i y_i x_i$
  2. $\frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$
* **Condizioni di Complementarietà KKT**:
  $$\alpha_i \big[ y_i(w^T x_i + b) - 1 \big] = 0$$
  *(Se $\alpha_i > 0$, il punto $x_i$ giace esattamente sul margine ed è un Vettore di Supporto).*

---

### Q13: *"Scrivi la Soft Margin SVM. Qual è il ruolo del parametro $C$?"*
* **Formulazione Primale**:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{sotto vincoli } y_i(w^T x_i + b) \ge 1 - \xi_i, \,\, \xi_i \ge 0$$
* **Ruolo del Parametro $C$**:
  * $C$ è il **fattore di penalizzazione degli errori** (trade-off tra ampiezza del margine e violazioni).
  * **$C \to \infty$**: Impone la Hard Margin SVM (tolleranza zero per le violazioni $\xi_i \to 0$, rischio overfitting).
  * **$C$ piccolo**: Tollera più violazioni $\xi_i > 0$, privilegiando l'ampliamento del margine $M = 2/\|w\|$ per prevenire l'overfitting.

---

### Q14: *"Scrivi la Formulazione Duale della SVM con gli $\alpha$ ottimi. Perché si usano gli $\alpha$?"*
* **Formulazione Duale**:
  $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{sotto vincoli } 0 \le \alpha_i \le C, \,\, \sum_{i=1}^N \alpha_i y_i = 0$$
* **Perché si usano i moltiplicatori $\alpha_i$ (Vantaggi del Duale)**:
  1. *Dipendenza esclusiva da prodotti scalari*: I dati compaiono solo sotto forma di prodotti scalari $x_i^T x_j$, consentendo di applicare direttamente il **Kernel Trick** $K(x_i, x_j) = \Phi(x_i)^T \Phi(x_j)$.
  2. *Sparsità della Soluzione*: Grazie alle condizioni KKT, solo per i pochi Vettori di Supporto si ha $\alpha_i > 0$. La predizione per un nuovo punto $x$ si calcola velocemente come:
     $$\mathbf{h(x) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right)}$$

---

### Q15: *"Scrivi la formula della distanza $d(x,y)$ indotta da un Kernel. Quali proprietà deve avere il Kernel per definire una distanza valida?"*
* **Distanza $d(x,y)$ in Spazio Kernel via Prodotto Scalare**:
  $$\mathbf{d(x, y)^2 = \|\Phi(x) - \Phi(y)\|^2 = K(x, x) + K(y, y) - 2 K(x, y)}$$
* **Proprietà che deve soddisfare la funzione Kernel $K(x, z)$**:
  Per il **Teorema di Mercer**, $K(x, z)$ deve essere una funzione di Kernel valida se e solo se è **Simmetrica** ($K(x,z) = K(z,x)$) e la sua matrice dei gramiani è **Definita Semipositiva**:
  $$\sum_{i=1}^N \sum_{j=1}^N c_i c_j K(x_i, x_j) \ge 0 \quad \forall c_i, c_j \in \mathbb{R}$$
  Questo garantisce che la distanza associata soddisfi la disuguaglianza triangolare e che punti con elevata similarità $K(x,y) \to K(x,x)$ abbiano distanza tendente a zero $d(x,y) \to 0$.

---

### Q16: *"Scrivi il Kernel RBF Gaussiano. Spiega la similitudine tra SVM RBF e K-NN al variare di $\sigma$ (o $\gamma$)!"*
* **Formula del Kernel RBF Gaussiano**:
  $$K(x, z) = \exp\left( -\frac{\|x - z\|^2}{2\sigma^2} \right) = \exp\left( -\gamma \|x - z\|^2 \right) \quad \text{con } \gamma = \frac{1}{2\sigma^2}$$
* **Similitudine tra SVM RBF e K-NN al variare di $\sigma$**:
  * **$\sigma \to 0$ ($\gamma \to \infty$)**: La campana gaussiana diventa un impulso di Dirac strettissimo attorno a ciascun punto di addestramento. L'SVR/SVM assegna un peso non nullo solo all'immediato punto più vicino, comportandosi in modo identico ad un **1-Nearest Neighbor (1-NN)** (errore di training zero, massima località, rischio overfitting).
  * **$\sigma \to \infty$ ($\gamma \to 0$)**: La campana gaussiana si appiattisce uniformemente a $1$ su tutto lo spazio, perdendo località e mediando globalmente tutti i dati.

---

### Q17: *"Cos'è la Support Vector Regression (SVR)? Scrivi la formula finale di $h(x)$ e la $\epsilon$-Insensitive Loss."*
* **La $\epsilon$-Insensitive Loss**:
  $$L_\epsilon(y, f(x)) = \max\big(0, |y - f(x)| - \epsilon\big) = \begin{cases} 0 & \text{se } |y - f(x)| \le \epsilon \\ |y - f(x)| - \epsilon & \text{altresì} \end{cases}$$
  *(Costruisce un tubo di tolleranza $\pm \epsilon$. Gli errori interni al tubo non vengono penalizzati).*
* **Formula Finale della Predizione di SVR**:
  $$\mathbf{h(x) = \sum_{i \in SV} (\alpha_i - \alpha_i^*) K(x_i, x) + b}$$
  dove $\alpha_i, \alpha_i^* \ge 0$ sono i moltiplicatori duali associati alle deviazioni sopra e sotto il tubo $\epsilon$.

---

# 📌 MODULO 3: Statistical Learning Theory (SLT), VC-Dimension & SRM

### Q18: *"Fornisci la definizione esatta della VC-Dimension ($h_{VC}$)."*
* **Definizione Formale**:
  La **VC-Dimension** (Dimensione Vapnik-Chervonenkis) di una classe di ipotesi $\mathcal{H}$ è la **MASSIMA cardinalità $N$** per cui esiste ALMENO UN insieme di $N$ punti shatterato (frammentato) da $\mathcal{H}$, ovvero per cui $\mathcal{H}$ può realizzare tutte le $2^N$ possibili dicotomie binaria.
  *(Se $\mathcal{H}$ può shatterare insiemi arbitrariamente grandi, allora $h_{VC} = \infty$).*

---

### Q19: *"Cos'è il principio di Structural Risk Minimization (SRM)? Disegna il grafico Errore / VC-Dimension (Slide 18 di SLT1). Cosa succede al Rischio Empirico nella Hard Margin SVM?"*
* **Principio di Structural Risk Minimization (SRM)**:
  Definisce una gerarchia di sotto-spazi di ipotesi annidati $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \dots \subset \mathcal{H}_k$ con VC-Dimension crescente $h_1 < h_2 < \dots < h_k$. Il principio SRM seleziona l'ipotesi $h^*$ che minimizza il bound sul rischio reale:
  $$R(h) \le R_{emp}(h) + \Omega(N, h_{VC})$$

* **Grafico ASCII Errore vs VC-Dimension**:
```
  Errore ^
         | \                                / Curva del Rischio Reale R(h)
         |  \                              /  (ha un minimo al punto di SRM)
         |   \                            /
         |    \                          /   Confidenza VC \Omega(N, h_VC)
         |     \                        /    (cresce all'aumentare della VC-dim)
         |      \                      / 
         |       \____________________/
         |        \                  /
         |         \________________/  Rischio Empirico R_emp(h)
         |                             (decresce all'aumentare della VC-dim)
         +----------------------------------------------------> VC-Dimension (h_VC)
```

* **Cosa succede alla curva del Rischio Empirico nella Hard Margin SVM?**:
  Nel caso della Hard Margin SVM (su dati linearmente separabili), per costruzione l'errore empirico è **esattamente UGUALE A ZERO ($R_{emp}(h) = 0$)**.
  Di conseguenza, il principio di minimizzazione del rischio strutturale SRM si riduce alla **pura minimizzazione del termine di confidenza VC $\Omega(N, h_{VC})$**, che equivale a minimizzare $\|w\|^2$ e massimizzare il margine geometrico $M = \frac{2}{\|w\|}$!

---

### Q20: *"Qual è il Teorema di Vapnik per la VC-Dimension delle SVM? Scrivi la formula completa!"*
* **Formula del Bound della VC-Dimension per SVM (Vapnik 1995)**:
  Per iperpiani a margine geometrico $\gamma$ definiti su punti contenuti in una sfera di raggio $R$ ($\|x\| \le R$), la VC-Dimension è limitata da:
  $$\mathbf{h_{VC} \le \min\left( D, \left\lceil \frac{R^2}{\gamma^2} \right\rceil \right) + 1}$$
* **Teorema del Bound di Generalizzazione di Vapnik**:
  $$R(h) \le R_{emp}(h) + \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$

---

### Q21: *"Scrivi la formula della Varianza nella decomposizione dell'errore atteso. Cosa rappresenta l'operatore $\mathbb{E}_D$?"*
* **Decomposizione dell'Errore Atteso (MSE)**:
  $$\mathbb{E}_{D,\epsilon}[(y - h_D(x))^2] = \text{Bias}(x)^2 + \text{Varianza}(x) + \sigma^2$$
* **Formula Esplicita della Varianza**:
  $$\text{Varianza}(x) = \mathbb{E}_D \left[ \big( h_D(x) - \bar{h}(x) \big)^2 \right] \quad \text{con } \bar{h}(x) = \mathbb{E}_D[h_D(x)]$$
* **Cosa rappresenta $\mathbb{E}_D$?**:
  Rappresenta il **Valore Atteso teorico calcolato rispetto a tutti i possibili Training Set $D$** di dimensione $N$ estraibili dalla distribuzione congiunta $P(x,y)$. Misura la sensibilità del modello alle fluttuazioni stocastiche del particolare dataset estratto.

---

# 📌 MODULO 4: Modelli Lineari, Perceptrone & Novikoff

### Q22: *"Come si realizza la porta logica NOT con 1 solo Perceptrone?"*
* **Definizione**: Input $x \in \{0, 1\}$, Target $y = 1 - x$. In etichette $\{-1, +1\}$: per $x=0 \to y=+1$; per $x=1 \to y=-1$.
* **Scelta Pesi e Bias**: Pesi $w = -2$, bias $b = +1$.
* **Verifica dell'Output $f(x) = \text{sign}(w \cdot x + b)$**:
  * Per $x = 0 \implies \text{sign}(-2(0) + 1) = \text{sign}(+1) = +1$ (Corretto).
  * Per $x = 1 \implies \text{sign}(-2(1) + 1) = \text{sign}(-1) = -1$ (Corretto).

---

### Q23: *"Enuncia e dimostra a grandi linee il Teorema di Convergenza del Perceptrone (Novikoff 1962)."*
* **Ipotesi di Separabilità**: Esiste un vettore ottimo $w^*$ con $\|w^*\|=1$ tale che $y_i (w^{*T} x_i) \ge \gamma > 0$ per ogni pattern $i$.
* **Passaggio 1 (Upper Bound su $\|w_k\|^2$)**:
  Ad ogni errore si applica $w_k = w_{k-1} + \eta y_i x_i \implies \|w_k\|^2 \le k \eta^2 R^2$ (con $R = \max_i \|x_i\|$).
* **Passaggio 2 (Lower Bound su $w_k^T w^*$)**:
  $w_k^T w^* = w_{k-1}^T w^* + \eta y_i (x_i^T w^*) \ge k \eta \gamma$.
* **Passaggio 3 (Disuguaglianza di Cauchy-Schwarz)**:
  $$(k \eta \gamma)^2 \le (w_k^T w^*)^2 \le \|w_k\|^2 \|w^*\|^2 \le k \eta^2 R^2 \implies k \le \frac{R^2}{\gamma^2}$$

---

# 📌 MODULO 5: Deep Learning, Fenomeni Moderni & Architetture Avanzate

### Q24: *"Spiega la differenza tra Bagging e Boosting. A cosa si collega il Boosting nel programma del corso?"*
* **Bagging (Bootstrap Aggregating)**: Addestra modelli complessi in **PARALLELO** su campioni di bootstrap per **ridurre la VARIANZA** (es. Random Forest).
* **Boosting (es. AdaBoost)**: Addestra modelli semplici (*weak learners*) in **SEQUENZA**, ripesando i pattern errati per **ridurre il BIAS**.
* **A cosa si collega il Boosting nel programma del corso?**:
  1. Si collega alla **Cascade Correlation** (entrambi sono algoritmi costruttivi che aggiungono componenti in sequenza per correggere i residui d'errore precedenti).
  2. Si collega al concetto di **Margine** nella SLT (Freund & Schapire hanno dimostrato che AdaBoost massimizza il margine di classificazione sulle combinazioni lineari di weak learners).

---

### Q25: *"Quali sono le differenze tra SOM (Self-Organizing Maps di Kohonen) e K-Means? Cos'è la Vector Quantization Loss?"*
* **Differenze tra SOM e K-Means**:
  1. *Struttura Topologica*: Il **K-Means non ha alcuna struttura di griglia** tra i centroidi dei cluster; il **SOM ha una griglia topologica 2D (o 1D)** tra i neuroni della mappa.
  2. *Regola di Aggiornamento*: Il K-Means aggiorna **esclusivamente il centroide del cluster vincente** $w_{c^*}$. Il SOM aggiorna **sia la BMU (Best Matching Unit) sia tutti i neuroni vicini nella griglia** tramite la funzione di vicinato gaussiana $h_{ic}(t) = \exp\left(-\frac{d_{griglia}(i,c)^2}{2\sigma(t)^2}\right)$.
  3. *Procedimento equivalente al K-Means nel SOM*: La fase competitiva per trovare la **BMU** $c^* = \arg\min_i \|x - w_i\|$ (che equivale al passo di assegnazione Expectation del K-Means o della Softmax).
* **Vector Quantization (VQ) Loss**:
  $$L_{VQ} = \sum_{i=1}^N \|x_i - w_{c(x_i)}\|^2$$
  dove $c(x_i)$ indica il prototipo/centroide assegnato al pattern $x_i$.

---

### Q26: *"Quali sono le 3 proprietà caratterizzanti delle Recurrent Neural Networks (RNN)?"*
1. **Stato Nascosto Ricorrente (Memoria Temporale)**:
   $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
   Mantiene una memoria dinamica del contesto passato lungo la sequenza temporale.
2. **Condivisione dei Pesi nel Tempo (Parameter Sharing)**:
   Le matrici pesi $W_{hh}, W_{xh}, W_{hy}$ rimangono identiche per ogni passo temporale $t=1 \dots T$, consentendo alla rete di processare sequenze di lunghezza arbitraria.
3. **Addestramento tramite BPTT (Backpropagation Through Time) e Gradient Clipping**:
   Srotola la rete nel tempo per calcolare le derivate; gestisce l'esplosione dei gradienti tramite il **Gradient Clipping** ($\|g\| \to \text{threshold} \frac{g}{\|g\|}$).

---

### Q27: *"Cos'è il fenomeno del Double Descent e perché accade?"*
* **Cos'è**: La classica curva a U Bias-Varianza mostra che l'errore di test prima scende e poi sale all'aumentare della complessità. Tuttavia, nel regime **fortemente sovra-parametrizzato** (dove il numero di pesi $|W|$ supera di gran lunga il numero di dati $N$), l'errore di test scende una **seconda volta** formando la curva del "Double Descent".
* **Perché accade**: Oltre la soglia di interpolazione ($|W| \ge N$), esistono infinite soluzioni pesi che azzerano l'errore di training. Gli ottimizzatori stocastici come SGD possiedono un **bias induttivo implicito** che seleziona la soluzione interpolante a **minima norma pesi $\|w\|$**, garantendo un'elevata regolarizzazione e bassa varianza.

---

### Q28: *"Cos'è la Lottery Ticket Hypothesis (Frankle & Carbin 2018)?"*
* **Enunciato**:
  Una rete neurale densa e casualmente inizializzata contiene una sotto-rete isolata (un *"biglietto della lotteria vincente"* / *winning ticket*) che, se addestrata da sola fin dall'inizio mantenendo le sue **stesse condizioni di inizializzazione originali**, è in grado di raggiungere prestazioni di accuratezza paragonabili alla rete neurale completa nello stesso numero di epoche.

---

### Q29: *"Spiega l'algoritmo Cascade Correlation (Fahlman & Lebiere 1990): Formula della Covarianza, Pro e Contro."*
* **Algoritmo Costruttivo**: Parte da una rete senza strati nascosti ed aggiunge neuroni uno alla volta.
* **Formula della Covarianza $S$**:
  Il neurone candidato $c$ viene addestrato per **massimizzare** la covarianza con l'errore residuo della rete:
  $$S = \sum_{k \in \text{Output}} \left| \sum_p (o_{p,c} - \bar{o}_c)(E_{p,k} - \bar{E}_k) \right|$$
* **Ascesa del Gradiente (Sign PLUS)**: $\Delta w_{c,j} = +\eta \frac{\partial S}{\partial w_{c,j}}$.
* **Pesi Congelati (*Frozen Weights*)**: Una volta inserito nella rete, i pesi in ingresso al nuovo neurone vengono **congelati per sempre**.
* **Pro**: Risolve il *moving target problem* e determina automaticamente la topologia della rete.
* **Contro**: Struttura profonda a cascata sequenziale lenta nell'inferenza.

---

### Q30: *"Spiega le Random Neural Networks / Reservoir Computing (Echo State Networks ESN): Equazioni e Pro/Contro."*
* **Equazione di Stato del Bacino (*Reservoir*)**:
  $$x(t+1) = \tanh\big( W_{in} u(t+1) + W_{res} x(t) \big)$$
* **Output di Readout Lineare**:
  $$y(t+1) = W_{out} x(t+1)$$
* **Procedura di Addestramento**: I pesi $W_{in}$ e $W_{res}$ sono inizializzati casualmente e **congelati permanentemente**. Si addestra **solo la matrice di output $W_{out}$** in forma analitica chiusa tramite regressione Ridge:
  $$W_{out} = (X^T X + \lambda I)^{-1} X^T Y$$
* **Pro**: Addestramento istantaneo senza Backpropagation numerica.
* **Contro**: Richiede un bacino $W_{res}$ molto grande per garantire la proprietà di memoria "Echo State".

---

### Q31: *"Spiega il Message Passing per Graph Neural Networks (GNN), la formula ricorsiva di NN4G ed il Readout Globale."*
* **Message Passing per GNN**:
  1. *AGGREGATE*: $m_v^{(k)} = \text{AGGREGATE}\left( \{ h_u^{(k-1)} : u \in N(v) \} \right)$
  2. *UPDATE*: $h_v^{(k)} = \text{UPDATE}\left( h_v^{(k-1)}, m_v^{(k)} \right)$
* **Formula Ricorsiva di NN4G (Neural Network for Graphs)**:
  $$h_v^{(k)} = f\left( W_1 x_v + \sum_{l=1}^{k-1} W_{2,l} \sum_{u \in N(v)} h_u^{(l)} \right)$$
* **Readout Globale su Grafi**:
  $$y_G = W_{out} \left( \sum_{v \in V} h_v \right)$$

---

### Q32: *"Cosa sono gli Autoencoder Undercomplete? Quali sono i vantaggi ed il contesto d'uso?"*
* **Definizione**: Reti neurali non supervisionate addestrate a ricostruire l'input $\|x - \hat{x}\|^2$ attraverso un layer nascosto centrale con **dimensione latente ristretta** ($\text{dim}(z) < \text{dim}(x)$).
* **Vantaggi e Contesto d'Uso**:
  1. *Compressione non lineare*: Costringe la rete ad apprendere le caratteristiche salienti ed invarianti dei dati (estrazione non lineare delle componenti principali).
  2. *Pre-training non supervisionato*: Inizializza i pesi di reti neurali molto profonde scartando il decoder dopo la fase non supervisionata.
