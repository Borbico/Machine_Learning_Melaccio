# 🌶️ I 7 Punti Spinosi delle Slide e i "Ganci" Teorici per Micheli
**Gruppo CDM25: Leonardo Celati — Damiano Degliotti — Daniele Melaccio**  
*(Analisi Critica delle 22 Slide del Report: I dettagli "vulnerabili" su cui il Professore potrebbe fare Domande Teoriche da Lode)*

---

## 📌 Mappa Rapida dei 7 Punti Spinosi nelle Slide

| # | Slide Coinvolte | Punto Spinoso / Dettaglio Nelle Slide | "Gancio" Teorico Probabile di Micheli |
| :-: | :--- | :--- | :--- |
| **1** | Slide 3, 11 | *Preprocessing all'interno dei Fold (No Data Leakage)* | Cos'è il Data Leakage? Perché scalare fuori dai fold altera il bias/varianza della Cross-Validation? |
| **2** | Slide 6 | *KNN pesato sulla distanza ha Training Error = 0 per costruzione* | Perché fa errore 0? Significa che ha la VC-Dimension infinita ($h_{VC}=\infty$)? |
| **3** | Slide 8 | *SGD con Momentum classico e Nesterov Acceleration (NAG)* | Qual è la differenza matematica tra Momentum classico di Polyak e Nesterov (NAG)? Scrivi le formule! |
| **4** | Slide 8, 10 | *BCEWithLogitsLoss ed Early Stopping sul rumore di MONK-3* | Perché usate `BCEWithLogits` invece di Sigmoide + CrossEntropy? Cos'è la stabilità numerica? |
| **5** | Slide 13, 21, 22 | *Sensibilità al Seed di Optuna ed Inizializzazione Pesi a Zero* | Perché la loss delle Reti Neurali è non convessa? Perché l'inizializzazione a zero impedisce la Symmetry Breaking? |
| **6** | Slide 15, 18 | *SVR RBF ha TR MEE 0.26 e Internal 18.16 ($C=100, \gamma=3$)* | Come interpreti il ruolo di $C$ e $\gamma$? Perché $\gamma=3$ crea questo forte gap di varianza? |
| **7** | Slide 16 | *Metrica $R^2$ (Coefficiente di Determinazione) non riportata nella NN* | Cos'è l'$R^2$? Può essere negativo? Cosa significa $R^2 = 0.65$? |

---

## 🎙️ Spiegazione Dettagliata dei 7 Punti Spinosi e Risposte da 30 e Lode

---

### 🌶️ PUNTO SPINOSO 1 (Slide 3 e 11): Preprocessing in-Fold vs Data Leakage
* **Cosa c'è scritto nelle Slide**:  
  *"CUP preprocessing was performed within each training fold to prevent data leakage."*
* **Il "Gancio" di Micheli**:  
  *"Mi spieghi esattamente cos'è il Data Leakage? Se aveste applicato lo StandardScaler a tutto il dataset prima di dividere in Fold, cosa sarebbe successo alle stime di Cross-Validation?"*
* **Risposta Modello da Lode**:  
  > *"Il **Data Leakage** (contaminazione dei dati) si verifica quando informazioni provenienti dai dati di validazione o di test penetrano indirettamente nel processo di addestramento del modello.*  
  > *Se avessimo applicato lo `StandardScaler` a tutto il dataset prima della K-Fold, lo scaler avrebbe calcolato la media globale $\mu$ e la deviazione standard $\sigma$ includendo i punti del validation fold.*  
  > *Di conseguenza, quando il modello veniva valutato sul validation fold, stava in realtà lavorando su dati già 'normalizzati' con informazioni della propria distribuzione. Questo produce una **stima ottimistica e distorta dell'errore (Selection Bias)**, creando un'illusione di alta accuratezza ed occultando la reale varianza del modello.*  
  > *Per questo motivo abbiamo inserito lo `StandardScaler` **all'interno della pipeline scikit-learn**, calcolando `fit_transform` esclusivamente sui dati di train di ciascun fold e `transform` sul validation fold."*

---

### 🌶️ PUNTO SPINOSO 2 (Slide 6): KNN Pesato sulla Distanza ed Errore di Training = 0
* **Cosa c'è scritto nelle Slide**:  
  *"With distance-weighted KNN, the training error might be zero by construction, as each training point is its own nearest neighbor at distance zero."*
* **Il "Gancio" di Micheli**:  
  *"Vedo che sulle slide riportate che con i pesi basati sull'inverso della distanza il KNN ha errore di training pari a ZERO per costruzione. Mi spiega la matematica di questo fatto? E dal punto di vista della Statistical Learning Theory, qual è la VC-Dimension del 1-NN o del KNN con pesi inversi?"*
* **Risposta Modello da Lode**:  
  > *"Matematicamente, nel KNN pesato sulla distanza il peso assegnato al $j$-esimo vicino è $w_j = \frac{1}{d(x, x_j)}$. Quando valutiamo la predizione su un punto $x_i$ appartenente al training set, la distanza del punto da se stesso è $d(x_i, x_i) = 0$. Il peso del punto tende all'infinito ($w_i \to \infty$), rendendo il contributo di tutti gli altri vicini del tutto trascurabile. Di conseguenza, il modello restituisce al 100% l'etichetta o il target esatto di $x_i$, annullando l'errore di training per costruzione.*  
  > *Dal punto di vista della SLT, per il $1$-NN (o per il KNN con pesi inversi alla distanza), la **VC-Dimension è INFINITA ($h_{VC} = \infty$)**.*  
  > *Infatti, la classe di ipotesi può memorizzare qualsiasi combinazione arbitraria di etichette per $N$ punti nello spazio (comportandosi come una Lookup Table pura con capacità illimitata). Aumentare $K$ (es. $K=3$ o $K=5$) con pesatura uniforme limita invece la capacità espressiva, riducendo la VC-Dimension effettiva ed agendo come regolarizzazione."*

---

### 🌶️ PUNTO SPINOSO 3 (Slide 8): Momentum Classico vs Nesterov Acceleration (NAG)
* **Cosa c'è scritto nelle Slide**:  
  *"Optuna-derived NNs use SGD with common momentum and Nesterov acceleration..."*
* **Il "Gancio" di Micheli**:  
  *"Avete usato l'accelerazione di Nesterov. Mi dice qual è la differenza concettuale e matematica tra il Momentum classico di Polyak e il Nesterov Accelerated Gradient (NAG)?"*
* **Risposta Modello da Lode**:  
  > *"La differenza risiede nel punto in cui viene calcolato il gradiente della funzione di errore:*  
  > *1. **Momentum Classico di Polyak**: Calcola il gradiente nella posizione corrente dei pesi $w(t)$ e poi aggiunge una frazione dello spostamento precedente (inerzia):*  
  > $$\Delta w(t) = \alpha \Delta w(t-1) - \eta \nabla E(w(t))$$  
  > *2. **Nesterov Accelerated Gradient (NAG — Look-Ahead Momentum)**: Applica prima lo spostamento teorico dell'inerzia $w' = w(t) + \alpha \Delta w(t-1)$ ('guarda in avanti') e calcola il gradiente **nel punto futuro $w'$**:*  
  > $$\Delta w(t) = \alpha \Delta w(t-1) - \eta \nabla E\big(w(t) + \alpha \Delta w(t-1)\big)$$  
  > *Il vantaggio di NAG è che se l'inerzia sta spingendo i pesi in una direzione che sta per salire lungo la valle dell'errore, il gradiente calcolato 'in avanti' agisce da freno preventivo, riducendo notevolmente le oscillazioni attorno al minimo."*

---

### 🌶️ PUNTO SPINOSO 4 (Slide 8 e 10): BCEWithLogitsLoss e Regolarizzazione in MONK-3
* **Cosa c'è scritto nelle Slide**:  
  *"BCEWithLogits loss... MONK-3 Regularization controls overfitting: hold-out validation MSE..."*
* **Il "Gancio" di Micheli**:  
  *"Perché in PyTorch si usa `BCEWithLogitsLoss` invece di mettere la Sigmoide nell'ultimo layer e poi usare la classica `BCELoss`? E su MONK-3 come avete controllato il rumore?"*
* **Risposta Modello da Lode**:  
  > *"Per motivi di **Stabilità Numerica** e di **Efficienza del Gradiente**.*  
  > *1. **Stabilità Numerica**: `BCEWithLogitsLoss` combina la funzione Sigmoide e la Binary Cross-Entropy in un'unica operazione numerica sfruttando il trick del Log-Sum-Exp. Evita fenomeni di underflow/overflow che si verificano quando si calcola prima $\sigma(z)$ e poi $\ln(\sigma(z))$ per valori molto grandi o molto piccoli di $z$ dove $\sigma(z) \to 0$ o $\sigma(z) \to 1$.*  
  > *2. **Semplificazione della derivata**: La derivata della Loss rispetto all'input netto $z$ dell'ultimo strato assume la forma pulita $\frac{\partial L}{\partial z} = \sigma(z) - y$, eliminando il termine di saturazione $\sigma'(z)$ a denominatore.*  
  >  
  > *Su **MONK-3**, che contiene un 5% di rumore nel training set, abbiamo controllato l'overfitting introducendo il **Weight Decay ($L_2$)** per penalizzare norme elevate pesi ed applicando l'**Early Stopping** basato sulla validation loss per interrompere l'addestramento prima che la rete memorizzasse le etichette rumorose."*

---

### 🌶️ PUNTO SPINOSO 5 (Slide 13, 21, 22): Non Convessità della Loss ed Inizializzazione Pesi
* **Cosa c'è scritto nelle Slide**:  
  *"Optuna sensitivity to training seed... We forced initializing all weights to zero... Final NNs used Kaiming-uniform..."*
* **Il "Gancio" di Micheli**:  
  *"Perché variando il TRAINING_SEED la ricerca di Optuna trova architetture diverse ma con prestazioni simili? E perché l'inizializzazione dei pesi a zero fallisce?"*
* **Risposta Modello da Lode**:  
  > *"1. **Non Convessità e Sensibilità al Seed**: La superficie di errore di una Rete Neurale è fortemente **non convessa**, caratterizzata da un numero elevatissimo di minimi locali equivalenti, punti di sella (saddle points) e simmetrie dei pesi. Modificare il seed cambia l'inizializzazione casuale iniziale dei pesi e l'ordine di estrazione dei mini-batch in SGD. La discesa del gradiente viene così instradata verso bacini di attrazione differenti nella superficie d'errore che tuttavia raggiungono valori di Loss molto simili.*  
  > *2. **Fallimento dell'Inizializzazione a Zero**: Inizializzare i pesi a zero impedisce la **Rottura della Simmetria (Symmetry Breaking)**. Tutti i neuroni di uno strato nascosto calcolano la stessa attivazione $o_j = f(0)$ e ricevono lo stesso segnale di errore $\delta_j$. Di conseguenza, tutti i pesi si aggiornano in modo identico, facendo comportare l'intero strato nascosto come un **singolo neurone equivalente**.*  
  > *Abbiamo quindi adottato l'inizializzazione **Kaiming Uniform (He)**, che mantiene la varianza dei gradienti costante $\text{Var}(w) = \frac{2}{n_{in}}$, ideale per attivazioni ReLU/GELU."*

---

### 🌶️ PUNTO SPINOSO 6 (Slide 15 e 18): SVR RBF (TR MEE 0.26 vs Internal 18.16) ed iperparametri $C$ e $\gamma$
* **Cosa c'è scritto nelle Slide**:  
  *"RBF-SVR C=100; γ=3; ε=0.1 (TR MEE 0.2599 / Internal 18.1632)..."*
* **Il "Gancio" di Micheli**:  
  *"Vedo che l'SVR RBF ha un errore di training bassissimo (0.26) ma in test fa 18.16. Come interpreta dal punto di vista teorico l'effetto di $C=100$ e $\gamma=3$ su questa discrepanza?"*
* **Risposta Modello da Lode**:  
  > *"Questa discrepanza è la classica manifestazione dell'**Overfitting causato da iperparametri ad alta capacità**:*  
  > *1. **Ruolo di $\gamma=3$**: Il parametro $\gamma$ determina il raggio di influenza del kernel RBF $K(x, z) = \exp(-\gamma \|x-z\|^2)$. Un valore elevato come $\gamma=3$ rende le campane gaussiane estremamente strette attorno a ciascun punto di addestramento.*  
  > *2. **Ruolo di $C=100$**: Il parametro $C$ è la costante di penalizzazione delle violazioni del margine. Un valore elevato come $C=100$ tollera pochissime violazioni $\xi_i$, costringendo l'SVR ad adattarsi quasi individualmente ad ogni campione di train.*  
  > *La combinazione di $\gamma=3$ e $C=100$ crea una superficie di risposta costituita da 'picchi gaussiani locali' isolati attorno ai dati di train, azzerando l'errore di addestramento (TR MEE 0.26) ma aumentando la Varianza del modello e facendo salire l'errore di validazione a 18.16.*  
  > *Tuttavia, la Nested CV ha dimostrato che la capacità di generalizzazione rimane comunque buona (17.09), motivo per cui l'SVR è stato mantenuto nell'Ensemble ma con un peso contenuto (16-17%)."*

---

### 🌶️ PUNTO SPINOSO 7 (Slide 16): Significato dell'Indice $R^2$ (Coefficiente di Determinazione)
* **Cosa c'è scritto nelle Slide**:  
  *"R²: KNN 0.6538, SVR 0.6559, NN Not reported in latest NN run."*
* **Il "Gancio" di Micheli**:  
  *"Cos'è l'indice $R^2$ che riportate nella Slide 16? Qual è la sua formula, cosa significa un valore di 0.65 e può assumere valori negativi?"*
* **Risposta Modello da Lode**:  
  > *"L'indice $R^2$ (**Coefficiente di Determinazione**) misura la proporzione della varianza totale dei dati che viene spiegata dal modello di regressione:*  
  > $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{MSE(\text{modello})}{MSE(\text{baseline\_media})}$$  
  > *1. **Significato di $R^2 = 0.65$**: Significa che il nostro modello (es. KNN) è in grado di spiegare circa il **65% della varianza complessiva dei target**, lasciando non spiegato il restante 35% attribuibile al rumore o a relazioni non catturate.*  
  > *2. **Può essere negativo? SÌ!** Se un modello effettua predizioni estremamente inaccurate (pegiori del semplice predire la media dei dati $\bar{y}$), il suo $MSE(\text{modello})$ supera la varianza totale dei dati, rendendo il rapporto $> 1$ e quindi $R^2 < 0$. Un $R^2$ negativo è un segnale inequivocabile di grave **underfitting** o di forte **overfitting** su dati out-of-sample."*
