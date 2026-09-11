# 🎤 Discorso Naturale e Guida Semplificata: Slide 18 - 22
**Presentatore: Daniele Melaccio — Gruppo CDM25**  
*(Spiegazione Semplice dei Concetti + Discorso Naturale e Aderente alle Lezioni di Micheli)*

---

## 💡 PARTE 1: Capire i Concetti delle Tue 5 Slide in Parole Semplici

Prima di imparare cosa dire, ecco la spiegazione elementare di cosa c'è in queste 5 slide:

### 📄 Slide 18: La Tabella dei Risultati Finali della CUP
* **Cosa c'è nella slide**: Una tabella che confronta i 3 modelli singoli (KNN, SVR RBF, Rete Neurale) e l'**Ensemble** (la loro combinazione).
* **I 3 tipi di errore che vedi**:
  1. *Training MEE*: L'errore sui dati su cui il modello si è addestrato. È sempre ottimistico.
  2. *OOF Validation MEE*: L'errore calcolato sui 400 dati di sviluppo in Cross-Validation (quando il modello non aveva visto quel blocco di dati).
  3. *Internal Test MEE*: L'errore sui 100 dati tenuti da parte fin dall'inizio e **mai visti prima**.
* **Il risultato chiave**: Il KNN da solo è il modello singolo migliore (MEE 16.32). Ma l'**Ensemble vince su tutti**, abbassando l'errore a **16.10 in CV** ed a **16.55 sul Test Set interno**.

---

### 📄 Slide 19: Come abbiamo scelto l'Ensemble e i Pesi
* **Cosa c'è nella slide**: La spiegazione delle nostre scelte metodologiche.
* **Le 3 cose da ricordare**:
  1. Abbiamo **escluso il LinearSVR** (che faceva MEE 25.51, troppo alto per essere utile) e lo abbiamo usato solo come baseline.
  2. I pesi dell'Ensemble sono: **73.8% KNN**, **16.1% SVR RBF** e **10.1% Rete Neurale**.
  3. **La regola di Micheli**: I pesi li abbiamo calcolati e **congelati PRIMA** di andare a valutare il Test Set finale, per evitare il *Data Leakage*.

---

### 📄 Slide 20: La Learning Curve dell'Ensemble
* **Cosa c'è nella slide**: Un grafico che mostra l'errore dell'Ensemble al variare del **numero di dati di training forniti ($N$)**.
* **Cosa significa**: C'è un piccolo gap tra la curva di training e quella di validazione (un po' di *overfitting residuo*). Ma la linea della validazione continua a scendere all'aumentare dei dati: significa che se avessimo avuto più dei 500 campioni disponibili, il modello avrebbe generalizzato ancora meglio!

---

### 📄 Slide 21: Discussione, Inizializzazione Pesi e Limitazioni
* **Cosa c'è nella slide**: I punti di forza e le limitazioni del progetto.
* **Punti chiave**:
  * **Inizializzazione pesi**: Se metti i pesi della rete neurale tutti a zero, la rete fallisce. Abbiamo usato l'inizializzazione **Kaiming Uniform**.
  * **Guadagno dell'Ensemble**: Migliora il KNN di circa l'1.5%, un guadagno modesto ma **costante e riproducibile**.
  * **Limitazione principale**: 500 campioni sono pochini per addestrare reti neurali molto profonde.

---

### 📄 Slide 22 (Appendice): Inizializzazione dei Pesi a Zero
* **Cosa c'è nella slide**: La dimostrazione grafica di cosa succede se si inizializzano i pesi a zero.
* **Concetto del corso**: Se i pesi sono zero, tutti i neuroni dello stesso strato calcolano lo stesso output ed il gradiente si aggiorna allo stesso modo. Non si "rompe la simmetria" (**Symmetry Breaking**) e i neuroni diventano copie identiche tra loro.

---

## 🗣️ PARTE 2: Il Tuo Discorso Naturale per l'Orale (Aderente alle Lezioni di Micheli)

Ecco un discorso fluido, semplice e perfettamente allineato al linguaggio del corso:

---

### 📊 SLIDE 18: CUP Ensemble: Final MEE Summary
*(Proietta la Slide 18)*

> *"Grazie. Passiamo ora alla valutazione comparativa finale dei nostri modelli sulla ML CUP.*  
>  
> *Come mostrato nella tabella della **Slide 18**, confrontiamo le prestazioni dei tre modelli singoli e dell'Ensemble finale.*  
> *Osservando i risultati sui 400 dati di sviluppo in Cross-Validation, il **KNN** si conferma il miglior modello individuale con un MEE di **16.32**.*  
> *L'**SVR con Kernel RBF** ottiene un MEE di **17.78**, mentre la **Rete Neurale** si attesta a **20.28**.*  
>  
> *Il risultato fondamentale è che l'**Ensemble finale** ottiene il punteggio di errore più basso in assoluto: scende a **16.10 Out-Of-Fold** e conferma questo miglioramento a **16.55 sul Test Set interno** di 100 campioni trattenuti."*

---

### 🏆 SLIDE 19: Final Model Choice: Frozen Weighted Ensemble
*(Passa alla Slide 19)*

> *"Nella **Slide 19** spieghiamo le scelte metodologiche del nostro predictor finale:*  
>  
> *1. **Composizione dell'Ensemble**: Abbiamo combinato il KNN, che rappresenta il nostro modello guida per l'interpolazione locale, con l'SVR RBF e la Rete Neurale che aggiungono informazioni complementari.*  
> *2. **Esclusione del LinearSVR**: Abbiamo escluso il modello lineare perché le sue prestazioni erano troppo basse (MEE ~25.51) e non portava alcun beneficio.*  
> *3. **Pesi congelati dalle predizioni OOF**: Minimizzando l'errore MEE sulle predizioni Out-Of-Fold dei 400 dati di sviluppo, abbiamo ricavato i pesi: **73.8% al KNN**, **16.1% all'SVR** e **10.1% alla Rete Neurale**.*  
> *4. **Controllo del Data Leakage**: La regola fondamentale che abbiamo seguito è che sia i modelli che i pesi dell'Ensemble sono stati **congelati prima** di effettuare l'unica valutazione sul Test Set interno e generare le predizioni per il Blind Test."*

---

### 📈 SLIDE 20: Ensemble Learning Curve
*(Passa alla Slide 20)*

> *"Nella **Slide 20** analizziamo il comportamento dell'Ensemble mediante una **Learning Curve in funzione della dimensione del dataset di addestramento**.*  
>  
> *Non trattandosi di un addestramento ad epoche, valutiamo come varia l'errore al crescere del numero di campioni forniti al modello.*  
> *Notiamo un gap tra la curva di training e quella di validazione, che indica un lieve overfitting residuo.*  
> *Tuttavia, la curva di validazione decresce in modo costante all'aumentare dei dati, suggerendo che con un dataset più grande di 500 campioni l'errore di generalizzazione si ridurrebbe ulteriormente."*

---

### 💡 SLIDE 21: Discussion: Findings and Limitations
*(Passa alla Slide 21)*

> *"Passando alla **Slide 21**, riassumiamo i risultati principali e le limitazioni:*  
>  
> *1. **Inizializzazione dei pesi**: Abbiamo sperimentato che inizializzare i pesi della rete neurale a zero porta a risultati degeneri. Per la rete finale abbiamo quindi adottato l'inizializzazione **Kaiming Uniform**.*  
> *2. **Guadagno dell'Ensemble**: Il miglioramento rispetto al solo KNN è di circa l'1.5% in Out-Of-Fold e dell'1.0% sul Test interno, dimostrandosi un guadagno costante e riproducibile.*  
> *3. **Limitazione principale**: Il vincolo principale del progetto risiede nelle dimensioni ridotte del dataset di sviluppo (500 esempi), che limitano l'addestramento di reti neurali più complesse."*

---

### 🔬 SLIDE 22 (Appendice): Uniform Weight Initialization
*(Passa alla Slide 22 se il professore chiede chiarimenti sull'inizializzazione)*

> *"Infine, nella **Slide 22 dell'Appendice**, mostriamo il dettaglio dell'esperimento con i pesi azzerati.*  
> *Inizializzare i pesi a zero impedisce la **rottura della simmetria (Symmetry Breaking)**: tutti i neuroni dello stesso strato calcolano la stessa attivazione e si aggiornano allo stesso modo, rendendo il layer nascosto del tutto equivalente ad un singolo neurone."*

---

## ❓ PARTE 3: Le 4 Domande d'Orale Spiegate in Modo Elementare

Ecco le 4 domande principali che Micheli potrebbe farti su queste slide, spiegate nel modo più semplice possibile:

---

### ❓ DOMANDA 1: *"Perché se metto i pesi di una rete neurale a zero la rete non impara? Cos'è la Symmetry Breaking?"*
* **Come spiegarlo in parole semplici**:  
  > *"Se metti tutti i pesi a zero, tutti i neuroni dello stesso strato ricevono lo stesso input (zero), calcolano la stessa uscita e ricevono lo stesso errore durante la Backpropagation.*  
  > *Di conseguenza, tutti i pesi si aggiornano della stessa identica quantità. I neuroni rimangono copie identiche tra loro e non si 'rompe la simmetria' (**Failure of Symmetry Breaking**). L'intero strato nascosto si comporta come se ci fosse un solo neurone.*  
  > *Usando invece l'inizializzazione **Kaiming Uniform**, i pesi partono da valori casuali ben bilanciati, permettendo a ciascun neurone di imparare caratteristiche diverse."*

---

### ❓ DOMANDA 2: *"Perché unire un modello forte (KNN) e modelli meno forti (SVR e Rete Neurale) migliora il risultato finale?"*
* **Come spiegarlo in parole semplici**:  
  > *"Perché i modelli appartengono a famiglie diverse e fanno **errori diversi su punti diversi**.*  
  > *Il KNN è un interpolatore locale basato sulla distanza, mentre la Rete Neurale approssima una funzione globale.*  
  > *Poiché i loro errori non sono correlati, quando facciamo la media pesata, l'errore di un modello viene compensato dal valore corretto dell'altro, riducendo l'errore complessivo dell'Ensemble a 16.10."*

---

### ❓ DOMANDA 3: *"Cosa sono le predizioni Out-Of-Fold (OOF) e perché le avete usate per calcolare i pesi?"*
* **Come spiegarlo in parole semplici**:  
  > *"Le predizioni Out-Of-Fold sono le predizioni generate da ciascun modello sui dati di validazione di ciascun fold (cioè dati che quel modello NON ha visto durante il suo addestramento).*  
  > *Abbiamo usato le predizioni OOF per trovare i pesi dell'Ensemble per evitare il **Data Leakage**.*  
  > *Se avessimo usato i dati di training, l'algoritmo avrebbe dato un peso enorme all'SVR solo perché l'SVR memorizza i dati di train, prendendo una solenne cantonata sull'overfitting."*

---

### ❓ DOMANDA 4: *"Perché sostenete che l'Internal Test Set di 100 campioni sia una stima non polarizzata (unbiased)?"*
* **Come spiegarlo in parole简单的**:  
  > *"Perché quei 100 campioni li abbiamo messi da parte all'inizio e **congelati completamente**.*  
  > *Non li abbiamo mai usati per scegliere gli iperparametri né per calcolare i pesi dell'Ensemble.*  
  > *Valutare l'Ensemble una sola volta su quei 100 dati mai visti ci ha dato la conferma reale che il modello generalizza bene (MEE 16.55) prima di generare il file per il Blind Test."*
