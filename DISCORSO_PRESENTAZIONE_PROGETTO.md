# 🗣️ Scaletta e Copione Presentazione Orale del Progetto
**Gruppo CDM25: Leonardo Celati — Damiano Degliotti — Daniele Melaccio**  
*(Suddivisione Integrale del Copione in 3 Relatori + Domande e Risposte Probabili del Prof. Micheli)*

---

## ⏱️ Scaletta e Suddivisione dei Ruoli (18 minuti complessivi)

| Relatore | Sezioni Assegnate | Argomenti Trattati | Durata |
| :--- | :--- | :--- | :--- |
| **LEONARDO** | Sezioni 1, 2, 3 | **Introduzione, Metodologia Sperimentale & Dataset MONK** (Pipeline, K-Fold, Data Leakage, MONK 1-2-3). | ~5-6 min |
| **DAMIANO** | Sezioni 4, 5, 6, 7 | **Dataset ML CUP, Preprocessing, Baseline LinearSVR & Rete Neurale PyTorch** (Loss MEE 4D, Gelu, AdamW, Optuna). | ~5-6 min |
| **DANIELE** | Sezioni 8, 9, 10 | **KNN, SVR RBF, Ensemble Finale e Conclusioni** (Overfitting vs Generalizzazione, pesi OOF 74.3%/17%/8.8%, MEE 16.10 OOF). | ~5-6 min |

---

## 🎤 Copione Dettagliato Diviso per Relatore

### 🟢 PARTE 1: Introduzione, Metodologia & MONK (Relatore: LEONARDO)

#### 1. Introduzione e Obiettivi
> *"Buongiorno. Nel nostro progetto abbiamo affrontato due problemi di apprendimento supervisionato con caratteristiche molto diverse.*  
> *La prima parte riguarda i **tre dataset MONK**, che rappresentano problemi di classificazione binaria su attributi categorici, particolarmente utili per analizzare il comportamento dei modelli in presenza di regole logiche, campioni ridotti, sbilanciamento delle classi e rumore.*  
> *La seconda parte riguarda la **ML CUP 2025**, un problema di regressione multi-output dove, a partire da 11 variabili di input, dobbiamo prevedere contemporaneamente 4 target continui.*  
> *L'obiettivo non è stato semplicemente trovare il modello con il miglior punteggio di training, ma costruire un **processo sperimentale rigoroso ed affidabile**, analizzando preprocessing, selezione degli iperparametri, K-Fold Cross-Validation, trade-off bias-varianza e costruzione dell'ensemble finale."*

#### 2. Metodologia Generale e Prevenzione del Data Leakage
> *"Abbiamo mantenuto una metodologia uniforme in tutti i notebook. Per la selezione degli iperparametri e del modello abbiamo utilizzato una **5-Fold Cross-Validation con mescolamento dei dati (shuffle)** per garantire che ogni fold rispecchiasse la distribuzione delle classi o del target.*  
> *Per la CUP abbiamo distinto una Cross-Validation standard per la valutazione finale ed una **Nested Cross-Validation** (con doppio ciclo interno/esterno) per stimare le prestazioni senza selection bias.*  
> *Abbiamo inoltre impiegato il **Bootstrap Out-Of-Bag (OOB)** per verificare la stabilità dei modelli.*  
> *Regola fondamentale di tutta la pipeline: **ogni trasformazione sui dati (come lo StandardScaler) è stata adattata ESCLUSIVAMENTE sui dati di training di ciascun fold**, per prevenire qualsiasi forma di Data Leakage."*

#### 3. Risultati sui Dataset MONK
> *"Sui tre problemi MONK abbiamo confrontato Reti Neurali (MLP con Backpropagation manuale e PyTorch), KNN e SVM con kernel polinomiale e RBF.*  
> * **MONK-1** (regola logica $(a_1=a_2)$ oppure $a_5=1$): Le reti neurali e le SVM con kernel polinomiale di grado 2 o 3 raggiungono il **100% di accuratezza sia in train che in test**, apprendendo perfettamente la regola logica. Il KNN fatica di più perché la similarità euclidea non cattura bene le relazioni booleane.*  
> * **MONK-2** (regola esatta "esattamente due $a_i=1$"): Essendo un problema di parità combinatoria, il KNN fallisce. Le **SVM con Kernel RBF Gaussiano** e le **Reti Neurali con strato nascosto (4-8 neuroni)** raggiungono il **100% di accuracy**, combinando le iper-superfici per racchiudere i pattern positivi.*  
> * **MONK-3** (regola logica con il 5% di rumore nel training set): Senza regolarizzazione la rete neurale va in overfitting sui pattern rumorosi. Introducendo **Weight Decay ($L_2$) ed Early Stopping**, la rete neurale e la SVM Soft Margin raggiungono **oltre il 97.2% di accuracy sul test set**, dimostrando la capacità di ignorare il rumore."*

---

### 🟡 PARTE 2: ML CUP, Preprocessing & Rete Neurale (Relatore: DAMIANO)

#### 4. Caratteristiche del Dataset ML CUP 2025
> *"Passando alla ML CUP 2025, il dataset contiene 500 esempi etichettati di sviluppo e 1000 esempi non etichettati di blind test. Ogni pattern ha 11 feature trasformate e 4 target continui.*  
> *La metrica ufficiale di gara è il **Mean Euclidean Error (MEE)**, ovvero la media della distanza euclidea vettoriale 4D tra il target reale e la predizione:*  
> $$L_{MEE} = \frac{1}{N} \sum_{i=1}^N \sqrt{\sum_{m=1}^4 (y_{i,m} - \hat{y}_{i,m})^2}$$  
> *La MEE considera congiuntamente l'errore sulle 4 uscite ed è la metrica coerente con la natura multi-output del problema. Come riferimento di baseline, un **DummyRegressor** che predice la media ottiene una MEE di **35.79**."*

#### 5. Baseline Lineare (LinearSVR)
> *"Come primo modello appreso abbiamo valutato il **LinearSVR** applicato separatamente ai 4 target tramite MultiOutputRegressor.*  
> *La configurazione migliore ($C=0.01$, loss squared epsilon-insensitive) ottiene una training MEE di **25.27** ed una Nested Validation MEE di **25.51**.*  
> *Il gap tra training e validation è minimo (0.24), dimostrando che il modello è molto stabile, ma affetto da **Underfitting**: la relazione tra input e target è fortemente non lineare e non può essere rappresentata da un iperpiano. LinearSVR è stato tenuto come baseline appresa ma escluso dall'ensemble."*

#### 6. Rete Neurale per la CUP (PyTorch + Optuna)
> *"Abbiamo poi costruito una **Rete Neurale di Regressione Multi-Output in PyTorch** con 4 neuroni lineari in uscita ed una Loss Function personalizzata basata direttamente sulla MEE.*  
> *Abbiamo ottimizzato l'architettura tramite **Optuna** e validato manualmente i risultati. La configurazione ottimale confluita nell'ensemble utilizza:*  
> * **2 strati nascosti da 128 e 64 neuroni** con attivazione **GELU**.  
> * Ottimizzatore **AdamW** (batch size 16), scheduler **ReduceLROnPlateau**, Early Stopping e Weight Decay.*  
> *Nei run dedicati la rete neurale ottiene una MEE di validazione di circa **20.28 Out-Of-Fold**. La rete migliora nettamente il modello lineare, ma sui soli 500 campioni disponibili fatica a raggiungere le prestazioni dei metodi basati su similarità. È stata comunque inserita nell'ensemble per la diversa natura funzionale delle sue predizioni."*

---

### 🔵 PARTE 3: KNN, SVM RBF, Ensemble e Conclusioni (Relatore: DANIELE)

#### 7. KNN e Support Vector Regression (SVR RBF)
> *"I due modelli individuali più competitivi per la CUP sono stati KNN ed SVR con kernel RBF:*  
> * **KNN (Miglior Modello Singolo)**: Inserito in pipeline con `StandardScaler`, la configurazione con $K=3$ vicini, pesi uniformi e distanza euclidea ottiene una **Nested MEE di 16.28** e un **Bootstrap OOB di 17.02**. Con un training error di 10.78, mostra il miglior bilanciamento tra bias e varianza.*  
> * **SVR RBF**: Applicando 4 SVR non lineari ($C=30, \gamma=3, \epsilon=0.1$), otteniamo una **Nested MEE di 17.09**. L'SVR RBF ha un training error bassissimo (~1.13) a causa dell'elevata località del kernel ($\gamma=3$), mostrandosi affetto da un marked overfitting, ma mantiene comunque un'ottima capacità di generalizzazione attorno a 17.09."*

#### 8. L'Ensemble Finale e Risultati del Blind Test
> *"Nell'ultima fase abbiamo costruito un **Ensemble Ottimizzato**:*  
> * Per garantire un'ulteriore verifica indipendente, abbiamo separato i 500 dati in **400 campioni di Development** e **100 campioni di Internal Test** mai usati per scegliere pesi o iperparametri.*  
> * Sui 400 campioni di development abbiamo estratto le predizioni Out-Of-Fold (OOF) dei 3 modelli (KNN, SVR RBF e Rete Neurale).*  
> * Abbiamo calcolato i pesi dell'ensemble minimizzando direttamente la MEE delle predizioni OOF con vincolo di non-negatività e somma 1. I pesi ottimali risultanti sono:*  
>   * **KNN: 74.3%**  
>   * **SVR RBF: 17.0%**  
>   * **Rete Neurale: 8.8%**  
> * **Risultati dell'Ensemble**:  
>   * MEE Out-Of-Fold (sui 400 dev): **16.10** (rispetto a 16.32 del solo KNN).  
>   * MEE sull'Internal Test (100 campioni mai visti): **16.55** (rispetto a 16.82 del solo KNN).  
> *Infine, abbiamo riaddestrato i 3 modelli sull'intero dataset di 500 campioni e combinato le predizioni sui 1000 esempi del Blind Test con i pesi congelati, generando il file finale `CDM25_ML-CUP25-TS.csv`."*

#### 9. Conclusioni del Progetto
> *"In conclusione, il progetto ha dimostrato che sui dataset discreti e logici come i MONK, le Reti Neurali e le SVM non lineari dominano. Sulla CUP, un problema di regressione continuo su soli 500 campioni, i metodi basati sulla similarità locale come il KNN offrono la generalizzazione migliore, mentre l'Ensemble finale beneficia della combinazione pesata di famiglie di modelli diverse per raggiungere il punteggio MEE più basso in assoluto."*

---

## ❓ Le 5 Domande Più Probabili di Micheli sul Progetto (con Risposte Modello)

### Q1: *"Perché avete usato il MEE invece dell'MSE per la CUP e come l'avete gestito in PyTorch?"*
* **Risposta Modello**: *"La MEE (Mean Euclidean Error) è la metrica ufficiale della ML CUP perché misura la reale distanza geometrica euclidea vettoriale 4D nello spazio dei target. In PyTorch abbiamo definito una Loss personalizzata `MEELoss` derivabile per il training. Per evitare singolarità nella derivata della radice quadrata in corrispondenza dell'origine ($\hat{y} = y$), abbiamo aggiunto un piccolo valore $\epsilon = 1e-8$ sotto radice ($\sqrt{\sum (y-\hat{y})^2 + \epsilon}$)."*

### Q2: *"Perché il KNN ha battuto la Rete Neurale sulla CUP?"*
* **Risposta Modello**: *"Il dataset della CUP ha soltanto 500 campioni di addestramento per 11 feature e 4 target. Le reti neurali profonde hanno una capacità esplicita elevata e richiedono migliaia di dati per sintonizzare centinaia di pesi senza overfittare. Il KNN invece è un learner instance-based non parametrico che sfrutta direttamente la continuità locale dei 500 punti nello spazio scalato, risultando meno soggetto a sovrastime di varianza su piccoli dataset."*

### Q3: *"Come avete evitato il Data Leakage durante la Cross-Validation e l'Ensemble?"*
* **Risposta Modello**: *"Abbiamo inserito lo `StandardScaler` all'interno delle pipeline scikit-learn o adattato lo scaler ESCLUSIVAMENTE sui fold di training di ciascun ciclo di K-Fold. Inoltre, per la scelta dei pesi dell'ensemble, abbiamo usato le predizioni Out-Of-Fold (OOF) sui 400 campioni di development, garantendo che i pesi venissero calcolati su predizioni generate da modelli che non avevano visto quei campioni in fase di train. Infine, abbiamo preservato 100 campioni di Internal Test completamente vergini per la stima finale."*

### Q4: *"Perché avete inserito la Rete Neurale nell'Ensemble se aveva una MEE peggiore (20.28 vs 16.32 di KNN)?"*
* **Risposta Modello**: *"Per il principio della diversità degli errori negli Ensemble: la rete neurale impara una funzione di regressione continua globale molto diversa dall'interpolazione locale del KNN. L'analisi di correlazione degli errori OOF ha mostrato che la rete neurale aveva la correlazione d'errore più bassa rispetto al KNN (0.702). Assegnarle un piccolo peso dell'8.8% ha aiutato l'Ensemble a compensare gli errori locali del KNN, riducendo la MEE complessiva a 16.10."*

### Q5: *"Qual è la differenza tra la vostra Validation K-Fold e la Nested Cross-Validation che avete implementato?"*
* **Risposta Modello**: *"La K-Fold standard valuta una specifica configurazione fissa di iperparametri. La Nested Cross-Validation (con ciclo interno a 3-5 fold per la GridSearch e ciclo esterno per l'assessment) permette di valutare l'intero algoritmo di selezione degli iperparametri su dati mai usati per la scelta degli stessi, fornendo una stima totalmente un-biased del rischio reale del modello."*
