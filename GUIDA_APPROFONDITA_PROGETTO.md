# 🔬 Guida Approfondita al Codice del Progetto e Previsione Osservazioni di Micheli
**Gruppo CDM25: Leonardo Celati — Damiano Degliotti — Daniele Melaccio**  
*(Dettagli Implementativi delle Classi Python, Architettura dell'Ensemble e le 6 Osservazioni Critiche di Micheli con Risposte da 30 e Lode)*

---

## 🛠️ PARTE 1: Le Scelte Implementative nel Codice Python

Il vostro codice è organizzato in una struttura **modulare, riutilizzabile e priva di duplicazioni**, suddivisa tra file di supporto `.py` e notebook d'esperimento `.ipynb`:

### 1. La Gestione Condivisa delle Split (`notebooks/cross_common.py`)
* **Classe `ManualSplitStrategy`**:  
  Invece di lasciare che ciascun notebook generi split casuali o disallineati, avete implementato una strategia di split manuale che garantisce che **tutti i modelli (KNN, SVR, Rete Neurale) vengano addestrati e valutati su ESATTAMENTE gli stessi 5 Fold**.
* **Perché è una scelta fondamentale**:  
  Garantisce la **confrontabilità diretta delle predizioni Out-Of-Fold (OOF)** necessarie per calcolare i pesi dell'Ensemble. Se i fold fossero stati diversi per ciascun modello, le predizioni OOF non avrebbero avuto corrispondenza campione per campione.
* **Classe `FoldResults`**:  
  Oggetto contenitore che raccoglie in modo strutturato train loss, validation loss, predizioni OOF, parametri e metriche (MSE, RMSE, MEE) per ciascun ciclo di Cross-Validation.

---

### 2. Gestione del Dataset ML CUP (`notebooks/cup_common.py`)
* **Splitting 400 Development / 100 Internal Test**:  
  Dei 500 esempi etichettati del file `ML-CUP25-TR.csv`, ne avete separati **400 per il Development Set** (usati per K-Fold, ricerca iperparametri ed ottimizzazione pesi ensemble) e **100 per l'Internal Test Set** (trattenuti e mai toccati durante la fase di design per il Model Assessment finale).
* **Definizione dello Scorer MEE per Scikit-Learn**:  
  Avete registrato il custom scorer `neg_mean_euclidean_error` compatibile con `GridSearchCV` e `RandomizedSearchCV`:
  $$L_{MEE} = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^4 (y_{i,m} - \hat{y}_{i,m})^2}$$

---

### 3. La Rete Neurale PyTorch + Optuna (`cup_NN_optuna.ipynb` & `cup_NN_manual.ipynb`)
* **Architettura**: Multi-Layer Perceptron di regressione con 11 input e 4 output lineari continuativi.
* **Loss Function Personalizzata `MEELoss`**:  
  Implementata in PyTorch con correzione numerica per la derivata della radice quadrata:
  $$\text{Loss}_{MEE} = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^4 (y_{i,m} - \hat{y}_{i,m})^2 + \epsilon} \quad \text{con } \epsilon = 1e-8$$
* **Iperparametri Ottimizzati con Optuna**:  
  - 2 strati nascosti da 128 e 64 neuroni con attivazione **GELU**.
  - Ottimizzatore **AdamW** con Weight Decay per la regolarizzazione $L_2$.
  - Schedulatore **`ReduceLROnPlateau`** che riduce il learning rate quando la validation MEE si stabilizza.
  - **`EarlyStopping`** con salvataggio dello stato dei pesi migliori (`best_epoch`).

---

### 4. Costruzione dell'Ensemble Ottimizzato (`cup_Ensemble.ipynb`)
* **Estrazione delle Predizioni OOF**:  
  Sui 400 campioni di development, ciascun modello (KNN, SVR RBF, Rete Neurale) ha generato un vettore di predizioni Out-Of-Fold $P_{knn}, P_{svr}, P_{nn} \in \mathbb{R}^{400 \times 4}$.
* **Calcolo dei Pesi Ottimali via Scipy `minimize`**:  
  I pesi $w = (w_{knn}, w_{svr}, w_{nn})$ sono stati ricavati risolvendo il problema di minimizzazione vincolata:
  $$\min_{w} \text{MEE}\Big( y_{dev}, \, w_{knn} P_{knn} + w_{svr} P_{svr} + w_{nn} P_{nn} \Big) \quad \text{sotto vincoli } w_i \ge 0, \,\, \sum_{i} w_i = 1$$
* **Pesi Risultanti**:
  * **KNN: 74.3%**
  * **SVR RBF: 17.0%**
  * **Rete Neurale: 8.8%**

---

## 🎯 PARTE 2: Previsione delle 6 Osservazioni Critiche di Micheli (con Risposte da 30 e Lode)

---

### ❓ Osservazione 1 di Micheli:
*"Ho notato che l'SVR RBF ha una MEE in training molto bassa (~1.13) ma in validazione fa ~17.09. C'è un forte gap. Il vostro modello è andato in Overfitting?"*

#### 💡 Come Rispondere (Risposta Modello):
> *"Sì professore, l'SVR RBF presenta un evidente gap tra train e validazione. Questo accade perché la ricerca degli iperparametri ha selezionato un valore di $\gamma = 3$, che rende il kernel RBF molto stretto e locale attorno ai singoli punti di train, unito ad un parametro $C = 30$ elevato che penalizza fortemente le violazioni.*  
> *Tuttavia, abbiamo verificato tramite la Nested Cross-Validation ed il Bootstrap Out-of-Bag che l'errore di generalizzazione si stabilizza attorno a 17.09 (molto vicino al 16.28 del KNN).*  
> *Proprio per via di questo gap di varianza, l'SVR RBF non è stato scelto come modello principale, ma gli è stato assegnato solo un peso contenuto del 17% nell'Ensemble per sfruttare il suo apporto complementare."*

---

### ❓ Osservazione 2 di Micheli:
*"La Rete Neurale sulla CUP ottiene una MEE di 20.28, nettamente peggiore rispetto al 16.28 del KNN. Perché ha prestazioni inferiori e perché le avete comunque assegnato l'8.8% di peso nell'Ensemble?"*

#### 💡 Come Rispondere (Risposta Modello):
> *"La Rete Neurale ha un'architettura da 128 e 64 neuroni ed è un approssimatore globale con molti parametri. Sui soli 500 campioni della CUP soffre della limitatezza dei dati rispetto a modelli basati su similarità locale come il KNN.*  
> *Tuttavia, la rete neurale impara una funzione di regressione continua globale di natura del tutto diversa dal KNN. L'analisi della matrice di correlazione degli errori Out-Of-Fold ha mostrato che gli errori della rete neurale erano quelli meno correlati con il KNN (correlazione 0.702).*  
> *L'ottimizzatore dell'Ensemble ha sfruttato questa diversità degli errori assegnando alla rete l'8.8% di peso, e questa integrazione ha permesso di abbattere la MEE Out-Of-Fold complessiva dell'Ensemble da 16.32 (del solo KNN) a 16.10."*

---

### ❓ Osservazione 3 di Micheli:
*"Sui problemi MONK, come spiegate il fatto che il KNN ottenga ottimi risultati in alcuni casi mentre su MONK-2 fallisce?"*

#### 💡 Come Rispondere (Risposta Modello):
> *"MONK-2 è definito dalla regola di parità combinatoria 'esattamente due attributi uguali ad 1'. La distanza euclidea tra vicini nello spazio delle feature categoriche non riflette questa struttura combinatoria: due punti geometricamente vicini possono avere parità opposte.*  
> *Modelli come le SVM con Kernel RBF e le MLP con strato nascosto riescono invece a combinare le iper-superfici per racchiudere la combinazione esatta, raggiungendo il 100% di accuracy, mentre il KNN fallisce per la natura della sua misura di distanza."*

---

### ❓ Osservazione 4 di Micheli:
*"Nel dataset MONK-3 c'era un 5% di rumore (etichette invertite) nel training set. Come avete impedito ai vostri modelli di memorizzare quel rumore?"*

#### 💡 Come Rispondere (Risposta Modello):
> *"Senza regolarizzazione, una rete neurale tenta di interpolare tutti i punti inclusi quelli rumorosi, andando in overfitting in train e fallendo in test.*  
> *Abbiamo introdotto la regolarizzazione di Tikhonov $L_2$ (Weight Decay) e l'Early Stopping basato sul Validation Set. L'Early Stopping interrompe l'addestramento prima che i pesi crescano abbastanza da memorizzare i pattern rumorosi isolati.*  
> *Grazie a questo, la rete neurale e la SVM Soft Margin hanno raggiunto oltre il 97.2% di accuracy sul Test Set, dimostrando di aver appreso la regola reale ignorando esattamente il 5% di rumore."*

---

### ❓ Osservazione 5 di Micheli:
*"Perché avete creato un Internal Test Set di 100 campioni separato oltre alla Cross-Validation sui 400 dati di development?"*

#### 💡 Come Rispondere (Risposta Modello):
> *"Per garantire un Model Assessment finale completamente non-polarizzato (unbiased).*  
> *Poiché le predizioni Out-Of-Fold sui 400 campioni di development sono state utilizzate direttamente per ottimizzare i pesi dell'Ensemble, la MEE di 16.10 poteva contenere un lieve selection bias dovuto all'ottimizzazione dei pesi.*  
> *Trattenere 100 campioni di Internal Test vergini ci ha permesso di valutare l'Ensemble una sola volta con pesi congelati su dati mai usati in alcuna fase, confermando una MEE di 16.55 e validando la reale capacità di generalizzazione del modello."*

---

### ❓ Osservazione 6 di Micheli:
*"Dal punto di vista matematico, la derivata della MEE contiene una radice quadrata a denominatore $\frac{1}{2\sqrt{z}}$. Come avete evitato che PyTorch crashasse per divisione per zero quando l'errore su un pattern è 0?"*

#### 💡 Come Rispondere (Risposta Modello):
> *"La funzione radice quadrata $\sqrt{z}$ ha derivata $\frac{1}{2\sqrt{z}}$ che tende all'infinito per $z \to 0$. Se durante l'addestramento la predizione coincide esattamente con il target ($\hat{y} = y$), la derivata esplode numericamente e genera valori NaN.*  
> *Nella nostra custom loss `MEELoss` in PyTorch abbiamo aggiunto una piccolissima costante di stabilizzazione $\epsilon = 1e-8$ sotto radice ($\sqrt{\sum (y - \hat{y})^2 + \epsilon}$), garantendo che il denominatore della derivata sia sempre strettamente positivo e continuo."*
