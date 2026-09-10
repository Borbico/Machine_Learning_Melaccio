# 🎤 Discorso Personalizzato per l'Orale: Slide 18 - 22
**Presentatore: Daniele Melaccio — Gruppo CDM25**  
*(Copione Dettagliato Parola per Parola + Guida Completa alle Domande Teoriche di Micheli sulle Slide 18-22)*

---

## 📌 Panoramica del Tuo Blocco (Grand Finale della Presentazione)

La tua parte è il **climax finale del progetto**: dopo che i tuoi compagni hanno introdotto la metodologia, i MONK e i singoli modelli CUP, tu presenti:
1. **Slide 18**: La **Tabella dei Risultati Finali** (confronto KNN, SVR RBF, Rete Neurale ed Ensemble).
2. **Slide 19**: La **Scelta del Modello Finale e Pesi Congelati** (perché l'Ensemble vince, esclusione del LinearSVR e pesi OOF).
3. **Slide 20**: Le **Curve di Apprendimento dell'Ensemble** al variare della dimensione del training set ($N$).
4. **Slide 21**: La **Discussione dei Risultati, Inizializzazione dei Pesi e Limitazioni**.
5. **Slide 22 (Appendice)**: La **Dimostrazione Teorica dell'Inizializzazione dei Pesi a Zero** (Symmetry Breaking).

---

## 🗣️ Copione Parola per Parola per l'Esposizione delle Slide

### 📊 SLIDE 18: CUP Ensemble: Final MEE Summary
*(Mostra la tabella dei risultati a schermo)*

> *"Grazie. Passiamo ora alla valutazione comparativa finale dei nostri modelli sulla ML CUP 2025.*  
> *Come mostrato nella **Slide 18**, la tabella riassume le prestazioni dei tre modelli individuali e dell'Ensemble finale rispetto a tre metriche:*  
> *1. Il **Training MEE** (misurato in-sample).*  
> *2. Il **Validation MEE Out-Of-Fold (OOF)** (calcolato sui 400 campioni di development).*  
> *3. L'**Internal Test MEE** (calcolato sui 100 campioni di test trattenuti e mai visti in fase di addestramento o selezione).*  
>  
> *Osservando i singoli modelli, confermiamo che il **KNN** è il miglior modello stand-alone con un OOF MEE di **16.32** ed un Internal Test di **16.82**.*  
> *L'**SVR RBF** mostra un forte gap tra training (~2.34) e validazione (~17.78), evidenziando un overfitting residuo dovuto alla località del kernel gaussiano.*  
> *La **Rete Neurale PyTorch** ottiene una MEE OOF di **20.28** e **21.75** sull'Internal Test, risentendo del numero limitato di campioni.*  
>  
> *Il risultato fondamentale è che l'**Ensemble** ottiene il punteggio MEE più basso in assoluto sia in Out-Of-Fold (**16.10**) sia sull'Internal Test (**16.55**), confermando che il guadagno di prestazione si trasferisce in modo coerente anche sul dataset di test indipendente."*

---

### 🏆 SLIDE 19: Final Model Choice: Frozen Weighted Ensemble
*(Passa alla Slide 19)*

> *"Nella **Slide 19** motiviamo la scelta del nostro predictor finale per la CUP:*  
> *Abbiamo scelto un **Ensemble pesato congelato** composto da KNN, SVR RBF e Rete Neurale.*  
>  
> *1. **Riconoscimento delle componenti**: Il KNN è il modello guida principale per la sua capacità di interpolazione locale. SVR RBF e Rete Neurale forniscono invece un contributo di informazione residua complementare.*  
> *2. **Esclusione del LinearSVR**: Il modello lineare è stato escluso dall'Ensemble ed impiegato solo come baseline, poiché le sue prestazioni (MEE ~25.51) erano troppo basse e non apportavano alcun beneficio.*  
> *3. **Pesi congelati dalle predizioni OOF**: Minimizzando la MEE Out-Of-Fold sui 400 campioni di dev con vincolo di non-negatività e somma 100%, abbiamo ricavato i pesi:*  
>    * **KNN: 73.84%**  
>    * **SVR RBF: 16.06%**  
>    * **Rete Neurale: 10.10%**  
> *4. **Controllo del Data Leakage**: Pesi ed iperparametri sono stati completamente **congelati prima** di effettuare l'unica valutazione sull'Internal Test e prima di generare le predizioni sul Blind Test.*  
> *5. **Addestramento Finale**: I tre modelli sono stati infine ri-addestrati sull'intero dataset di 500 campioni etichettati e combinati con i pesi congelati per produrre il file di consegna `CDM25_ML-CUP25-TS.csv`."*

---

### 📈 SLIDE 20: Ensemble Learning Curve
*(Passa alla Slide 20)*

> *"Nella **Slide 20** analizziamo il comportamento dell'Ensemble mediante una **Learning Curve in funzione della dimensione del dataset di addestramento (Training Set Size)**.*  
>  
> *Poiché l'Ensemble non ha un processo di addestramento ad epoche, abbiamo valutato l'errore MEE di training e di validazione al crescere del numero di dati forniti ai modelli.*  
> *Notiamo un **gap persistente** tra la curva di training e quella di validazione, il che conferma la presenza di un lieve overfitting residuo.*  
> *Tuttavia, all'aumentare dei dati di addestramento, la **Validation MEE decresce in modo costante e la sua variabilità (ampiezza dei fold) si stringe**, suggerendo che la disponibilità di un dataset più ampio ridurrebbe ulteriormente il gap di generalizzazione."*

---

### 💡 SLIDE 21: Discussion: Findings and Limitations
*(Passa alla Slide 21)*

> *"Passando alla **Slide 21**, discutiamo i principali risultati e le limitazioni del progetto:*  
>  
> *1. **Inizializzazione dei Pesi**: Abbiamo condotto esperimenti comparativi tramite la funzione `init_weights`. L'inizializzazione forzata a zero ha causato un comportamento di addestramento degenere ed instabile per la mancata rottura della simmetria. Per la rete neurale finale abbiamo quindi adottato l'inizializzazione **Kaiming Uniform (He Uniform)** con bias azzerati.*  
> *2. **Guadagno dell'Ensemble**: Il miglioramento dell'Ensemble rispetto al solo KNN è contenuto (circa **1.5% in OOF** e **1.0% sull'Internal Test**), ma è estremamente **costante e riproducibile**.*  
> *3. **Complementarità**: SVR e Rete Neurale aiutano l'Ensemble perché i loro residui d'errore sono poco correlati con quelli del KNN.*  
> *4. **Limitazioni principali**: Il limite principale risiede nelle dimensioni ridotte del dataset di sviluppo (500 esempi) che pongono un vincolo alla capacità di addestramento delle reti neurali profonde."*

---

### 🔬 SLIDE 22 (Appendice): Uniform Weight Initialization & Symmetry Breaking
*(Passa alla Slide 22 se viene richiesta una precisazione sull'inizializzazione)*

> *"Infine, nella **Slide 22 dell'Appendice**, mostriamo il dettaglio dell'esperimento sull'**inizializzazione dei pesi a zero**.*  
> *Inizializzare tutti i pesi ad un valore costante o nullo impedisce la **rottura della simmetria (Symmetry Breaking)**: tutti i neuroni dello stesso strato calcolano la stessa attivazione e ricevono lo stesso gradiente, comportandosi come un singolo neurone equivalente e bloccando l'apprendimento delle feature.*  
> *Il grafico evidenzia come la norma del gradiente si appiattisca e la loss rimanga bloccata su MONK-1."*

---

## ❓ Le 5 Domande Teoriche da Lode di Micheli collegate alle tue Slide (con Risposte Modello)

---

### ❓ DOMANDA 1 (Dalla Slide 21/22):
*"Mi parli dell'inizializzazione dei pesi nelle Reti Neurali. Perché se imposto tutti i pesi a zero la rete non impara? Qual è il termine teorico che cerco e come funziona l'inizializzazione di Kaiming/He?"*

#### 💡 Risposta Modello:
> *"Il termine teorico è la **Mancata Rottura della Simmetria (Failure of Symmetry Breaking)**.*  
> *Se tutti i pesi $w_{ji}$ di un layer nascosto sono inizializzati a zero:*  
> *1. Ogni neurone $j$ calcolerà lo stesso input netto $net_j = \sum 0 \cdot x_i + 0 = 0$ e la stessa attivazione $o_j = f(0)$.*  
> *2. Durante la Backpropagation, tutti i neuroni del layer riceveranno lo stesso segnale di errore $\delta_j$.*  
> *3. Di conseguenza, le derivate $\frac{\partial E}{\partial w_{ji}}$ saranno tutte identiche e tutti i pesi si aggiorneranno del medesimo valore $\Delta w$.*  
> *I neuroni rimarranno simmetrici per sempre, rendendo il layer nascosto del tutto equivalente ad un **singolo neurone** e distruggendo la capacità della rete.*  
>  
> *Per le reti con attivazione **ReLU/GELU** usiamo l'inizializzazione **Kaiming / He Uniform**, che estrae i pesi da una distribuzione uniforme $\mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$. Questa varianza controllata mantiene costante la varianza dei segnali di attivazione e dei gradienti attraverso i layer profondi, evitando l'esplosione o la scomparsa del gradiente."*

---

### ❓ DOMANDA 2 (Dalla Slide 18/19):
*"Perché combinare un modello buono (KNN con MEE 16.32) ed un modello mediocre (Rete Neurale con MEE 20.28) migliora il risultato finale a 16.10? Qual è il principio teorico degli Ensemble?"*

#### 💡 Risposta Modello:
> *"Il principio teorico alla base degli Ensemble è l'**Indipendenza e De-correlazione degli Errori**.*  
> *Quando combiniamo due stimatori $h_1(x)$ e $h_2(x)$, la varianza dell'errore dell'Ensemble dipende non solo dalle varianze dei singoli modelli, ma anche dalla loro **covarianza degli errori** $\text{Cov}(e_1, e_2)$.*  
> *Se i modelli appartengono a famiglie funzionali diverse (KNN è un interpolatore locale basato su distanza; la Rete Neurale è un approssimatore non lineare globale), i loro errori tendono ad avere segno opposto su diversi campioni.*  
> *Nel nostro progetto, l'analisi di correlazione ha mostrato che gli errori della rete neurale erano quelli meno correlati con il KNN (correlazione 0.702). L'algoritmo di minimizzazione ha assegnato il 10.1% di peso alla rete neurale proprio per annullare i residui locali del KNN, abbattendo la MEE globale a 16.10."*

---

### ❓ DOMANDA 3 (Dalla Slide 18/19):
*"Perché avete usato predizioni Out-Of-Fold (OOF) per calcolare i pesi dell'Ensemble anziché usare le predizioni di training?"*

#### 💡 Risposta Modello:
> *"Per evitare il **Selection Bias** ed il **Data Leakage nei pesi dell'Ensemble**.*  
> *Se avessimo calcolato i pesi minimizzando l'errore sui dati di training, l'ottimizzatore avrebbe assegnato un peso eccessivo all'SVR RBF, poiché l'SVR ha un training error bassissimo (~2.34) dovuto all'overfitting.*  
> *Utilizzando le predizioni Out-Of-Fold (OOF) sui 400 campioni di development, ogni predizione usata dall'ottimizzatore proviene da un fold in cui quel modello NON ha visto quel campione in fase di train. Questo garantisce che i pesi rispecchino la reale capacità di generalizzazione e non l'overfitting di addestramento."*

---

### ❓ DOMANDA 4 (Dalla Slide 20):
*"Che differenza c'è tra la Learning Curve della Slide 20 e la classica Learning Curve di addestramento di una Rete Neurale?"*

#### 💡 Risposta Modello:
> *"La classica Learning Curve di una Rete Neurale traccia la Loss in funzione delle **Epoche di addestramento** (con lo scopo di monitorare l'overfitting temporale ed individuare il punto di interruzione dell'Early Stopping).*  
> *La Learning Curve della **Slide 20** traccia la Loss MEE in funzione della **Dimensione del Dataset di Addestramento ($N$)**.*  
> *Serve per analizzare l'effetto dell'aumento dei campioni disponibili: il fatto che la curva di validazione decresca in modo costante ed assottigli la propria ampiezza all'aumentare di $N$ dimostra che l'overfitting residuo del nostro Ensemble diminuirebbe disponendo di un dataset di sviluppo più grande di 500 campioni."*

---

### ❓ DOMANDA 5 (Dalla Slide 18):
*"Perché sostenete che l'Internal Test Set di 100 campioni sia la vostra stima primaria di generalizzazione non polarizzata?"*

#### 💡 Risposta Modello:
> *"Perché i 100 campioni dell'Internal Test sono stati **trattenuti fin dall'inizio e completamente congelati**.*  
> *Non sono mai stati utilizzati per la ricerca degli iperparametri (GridSearch/Optuna) né per il calcolo dei pesi dell'Ensemble.*  
> *Mentre il valore MEE OOF di 16.10 poteva contenere un piccolissimo bias derivante dall'ottimizzazione dei pesi dell'Ensemble, valutare l'Ensemble una sola volta sui 100 campioni vergini ha confermato una MEE di 16.55, fornendo una stima un-biased della capacità di generalizzazione sul Blind Test reale."*
