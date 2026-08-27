# ❓ Compendio Integrale Quiz d'Esame — Machine Learning
**Corso del Prof. Alessio Micheli — Università di Pisa**
*(Raccolta completa, revisionata nell'ortografia ed arricchita con spiegazioni teorico-matematiche approfondite per tutte le 128 domande dei quiz)*

---

### 📌 Domanda 1 (ID: 1)
**Testo**: The gradient can be exploited to provide the direction toward local maxima of a function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il gradiente $\nabla f(x)$ indica la direzione di MASSIMA CRESCITA (ripidezza positiva) di una funzione. Pertanto, seguire la direzione del gradiente consente di muoversi verso un massimo locale (Gradient Ascent).

---

### 📌 Domanda 2 (ID: 2)
**Testo**: The negative of gradient can be exploited to provide the direction toward local minima of a function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'opposto del gradiente $-\nabla f(x)$ indica la direzione di MASSIMA DECRESCITA (ripidezza negativa) di una funzione. Seguire questa direzione è la base dell'algoritmo di Discesa del Gradiente (Gradient Descent) per raggiungere i minimi locali della loss function.

---

### 📌 Domanda 3 (ID: 3)
**Testo**: $$\|x\|_2 = \sum_i x_i^2$$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La norma Euclidea (o norma $L_2$) di un vettore $x$ è definita come la radice quadrata della somma dei quadrati delle componenti: $\|x\|_2 = \sqrt{\sum_i x_i^2}$. La formula indicata senza radice rappresenta la norma al quadrato $\|x\|_2^2$, quindi l'uguaglianza diretta è FALSO.

---

### 📌 Domanda 4 (ID: 4)
**Testo**: $\|x\|_2^2 = \sum_i x_i^2$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La norma Euclidea al quadrato $\|x\|_2^2$ coincide esattamente con il prodotto scalare del vettore con se stesso $x^T x = \sum_i x_i^2$.

---

### 📌 Domanda 5 (ID: 5)
**Testo**: There exists the search inductive bias

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Esistono due tipi fondamentali di Inductive Bias (bias induttivo): il **Language Bias** (o Restriction Bias, legato alla scelta dello spazio delle ipotesi $\mathcal{H}$) e il **Search Bias** (o Preference Bias, legato al modo in cui l'ottimizzatore esplora $\mathcal{H}$, es. Occam's Razor).

---

### 📌 Domanda 6 (ID: 6)
**Testo**: An unbiased learner corresponds to a look-up table model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un learner privo di bias induttivo (Unbiased Learner) può classificare univocamente solo i punti già visti nel Training Set (comportandosi come una tabella di consultazione / Lookup Table). Non ha alcuna base razionale per generalizzare su nuovi punti non visti.

---

### 📌 Domanda 7 (ID: 7)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the model response and its inputs

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nell'apprendimento supervisionato la Loss Function misura la discrepanza o 'distanza' tra la risposta del modello $h(x)$ e il target reale $y$ (non rispetto agli input $x$).

---

### 📌 Domanda 8 (ID: 8)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the target and the model inputs

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Loss Function valuta la distanza tra l'output predetto dal modello $h(x)$ e l'etichetta target $y$ (non tra il target e gli input).

---

### 📌 Domanda 9 (ID: 9)
**Testo**: Removing the inductive bias from a learning system brings advantages for the learning system

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Rimuovere il bias induttivo rende impossibile la generalizzazione su dati futuri non visti. Il bias induttivo è una condizione NECESSARIA per il Machine Learning.

---

### 📌 Domanda 10 (ID: 10)
**Testo**: Polynomial coefficients (magnitude) typically decreases after training with higher values of M

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Aumentando il grado $M$ di un polinomio, il modello diventa più complesso e tenta di interpolare perfettamente tutti i punti di train. Ciò causa un aumento notevole della magnitudo dei coefficienti polinomiali (fenomeno tipico dell'overfitting), non una diminuzione.

---

### 📌 Domanda 11 (ID: 11)
**Testo**: Polynomial coefficients (magnitude) typically increases after training with higher values of M

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Aumentando la complessità $M$, i coefficienti del polinomio crescono in magnitudo per seguire le fluttuazioni ad alta frequenza del rumore nel training set.

---

### 📌 Domanda 12 (ID: 12)
**Testo**: The model selection is made looking to best result on the test set

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Model Selection (scelta degli iperparametri e dell'architettura) deve essere effettuata ESCLUSIVAMENTE sul Validation Set, mai sul Test Set (che deve rimanere incontaminato per la stima finale del rischio).

---

### 📌 Domanda 13 (ID: 13)
**Testo**: The model selection is made looking to best result on the validation set

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Validation Set è lo strumento dedicato alla scelta del modello migliore e all'ottimizzazione degli iperparametri.

---

### 📌 Domanda 14 (ID: 14)
**Testo**: The model selection is made looking to best result on the training set

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Selezionare il modello in base all'errore di Training porterebbe a scegliere sempre il modello più complesso e overfitato.

---

### 📌 Domanda 15 (ID: 15)
**Testo**: The number of free parameters after a LBE is typically greater than in the original model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Linear Basis Expansion (LBE) proietta gli input originali $n$-dimensionali in uno spazio di feature $K$-dimensionale con $K > n$, aumentando il numero di parametri liberi $w_k$.

---

### 📌 Domanda 16 (ID: 16)
**Testo**: A linear classifier with LBE cannot solve a non-linear separable problem

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un modello lineare applicato a feature trasformate tramite LBE $h(x) = \sum w_k \phi_k(x)$ può definire confini di decisione non lineari nello spazio di input originale, risolvendo problemi non linearmente separabili.

---

### 📌 Domanda 17 (ID: 17)
**Testo**: A polynomial expansion is a form of LBE

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'espansione polinomiale (es. inserire $x_1^2, x_2^2, x_1 x_2$) è una forma classica di Linear Basis Expansion (LBE).

---

### 📌 Domanda 18 (ID: 18)
**Testo**: The perceptron is a LTU

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Perceptrone (Rosenblatt 1958) è un'Unità a Soglia Lineare (Linear Threshold Unit - LTU) avente la funzione a gradino $\text{sign}(z)$ come attivazione.

---

### 📌 Domanda 19 (ID: 19)
**Testo**: The perceptron learning algorithm works with a unit that compute $$sign(w^Tx)$$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'output del Perceptrone per un pattern $x$ con bias incorporato è calcolato come $o = \text{sign}(w^T x)$.

---

### 📌 Domanda 20 (ID: 20)
**Testo**: The perceptron learning algorithm cannot solve the XOR problem

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Perceptrone singolo limita i confini di decisione ad iperpiani lineari e pertanto non può separare il problema logico dell'XOR.

---

### 📌 Domanda 21 (ID: 21)
**Testo**: A small networks with a hidden layer of perceptrons can solve the XOR problem

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Una rete neurale a 2 strati con un layer nascosto di perceptroni può combinare più iperpiani per risolvere il problema dell'XOR.

---

### 📌 Domanda 22 (ID: 22)
**Testo**: The perceptron learning algorithm always converges in a finite number of steps

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Teorema di Convergenza del Perceptrone (Novikoff 1962) garantisce la convergenza in un numero finito di passi $k \le \frac{R^2}{\gamma^2}$ sotto la fondamentale assunzione che il dataset sia linearmente separabile con margine $\gamma > 0$.

---

### 📌 Domanda 23 (ID: 23)
**Testo**: The perceptron learning algorithm solves only tasks with all positive patterns

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'algoritmo del Perceptrone funziona con esempi sia positivi (+1) che negativi (-1).

---

### 📌 Domanda 24 (ID: 24)
**Testo**: The sigmoidal logistic is not a differentiable function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La funzione Sigmoide $\sigma(z) = \frac{1}{1+e^{-z}}$ è una funzione continua e derivabile ovunque, con derivata $\sigma'(z) = \sigma(z)(1-\sigma(z))$.

---

### 📌 Domanda 25 (ID: 25)
**Testo**: A linear model can be obtained by an artificial neural unit with an identity activation function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Usando l'attivazione Identità $f(x) = x$, l'uscita dell'unità è una combinazione lineare pura degli input $w^T x + b$, ottenendo un modello lineare.

---

### 📌 Domanda 26 (ID: 26)
**Testo**: The perceptron has a differentiable activation function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La funzione di attivazione a gradino del Perceptrone presenta una discontinuità a salto in zero e derivata nulla quasi ovunque, rendendola non derivabile in senso classico per il gradiente.

---

### 📌 Domanda 27 (ID: 27)
**Testo**: A NN can be seen a LBE only when we consider LBE for classification

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Le Reti Neurali agiscono come Linear Basis Expansion (LBE) sia nei compiti di classificazione che in quelli di regressione.

---

### 📌 Domanda 28 (ID: 28)
**Testo**: A NN can be seen as a LBE, and we showed it when the phi function of the LBE is the NN’s output neurons function

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nelle Reti Neurali viste come LBE, le funzioni di base $\phi_j(x)$ sono le uscite dei neuroni NASCOSTI ($\phi_j(x) = f_{hidden}(\sum w_{ji} x_i)$), non dei neuroni di output.

---

### 📌 Domanda 29 (ID: 29)
**Testo**: A NN cannot address a regression task, but only classification tasks

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Le Reti Neurali sono approssimatori universali sia per compiti di classificazione binaria/multiclasse che per la regressione continua.

---

### 📌 Domanda 30 (ID: 30)
**Testo**: The backpropagation is based on the minimization of the MSE

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Backpropagation è un algoritmo generico di calcolo del gradiente (tramite Chain Rule) applicabile a qualsiasi loss function derivabile (MSE, BCE, MEE, Cross-Entropy), non solo all'MSE.

---

### 📌 Domanda 31 (ID: 31)
**Testo**: The backpropagation allows us to obtain the update rule for the units finding the delta for each of them

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Backpropagation calcola l'errore locale $\delta_t$ per ogni neurone $t$, permettendo di ricavare la regola di aggiornamento pesi $\Delta w_{tu} = \eta \delta_t o_u$.

---

### 📌 Domanda 32 (ID: 32)
**Testo**: High values of w initialization accelerate the training

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Inizializzare i pesi con valori elevati porta le attivazioni sigmoidali/tanh in saturazione ($|z| > 4$), azzerando le derivate e bloccando l'apprendimento (Vanishing Gradient).

---

### 📌 Domanda 33 (ID: 33)
**Testo**: All zero values to initialize the weights can be a good strategy

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Inizializzare tutti i pesi a zero distrugge la simmetria della rete: tutti i neuroni dello strato nascosto calcolano lo stesso output e ricevono lo stesso gradiente, rimanendo identici.

---

### 📌 Domanda 34 (ID: 34)
**Testo**: It is always convenient to have a high eta value to speed-up the training

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un valore di learning rate $\eta$ troppo alto causa oscillazioni instabili o divergenza attorno al minimo della superficie d'errore.

---

### 📌 Domanda 35 (ID: 35)
**Testo**: The momentum has been introduces to provide a regularization of the model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Momentum $\alpha \Delta w(t-1)$ è introdotto per accelerare la convergenza e smorzare le oscillazioni nelle direzioni a forte curvatura, non come tecnica di regolarizzazione.

---

### 📌 Domanda 36 (ID: 36)
**Testo**: Too few hidden units can lead to underfitting

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un numero insufficiente di unità nascoste riduce eccessivamente la capacità espressiva del modello, portando all'Underfitting.

---

### 📌 Domanda 37 (ID: 37)
**Testo**: With a well regularized approach, early stopping can even be not necessary or it can be used without showing its effect

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Se si applica una forte regolarizzazione (es. Weight Decay $L_2$), la norma dei pesi rimane controllata e l'Early Stopping potrebbe non attivarsi affatto poiché la loss di validazione non aumenta.

---

### 📌 Domanda 38 (ID: 38)
**Testo**: In the Nesterov Momentum approach the gradient is computed after the movement due to the momentum

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nel Nesterov Accelerated Gradient (NAG), si applica prima lo spostamento dovuto al momentum $w' = w + \alpha \Delta w_{old}$ e si calcola il gradiente nel punto futuro $w'$.

---

### 📌 Domanda 39 (ID: 39)
**Testo**: In the cascade correlation algorithm the hidden units are trained to minimize their LMS

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nella Cascade Correlation, i candidati vengono addestrati per MASSIMIZZARE la covarianza $S$ con l'errore residuo della rete, non per minimizzare l'MSE.

---

### 📌 Domanda 40 (ID: 40)
**Testo**: The use of a K-fold CV can solve in itself the problem of a rigorous approach to the model selection and assessment

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La K-Fold CV fornisce una stima dell'errore, ma una rigida selezione e valutazione richiede l'architettura della Nested Cross-Validation per evitare il selection bias.

---

### 📌 Domanda 41 (ID: 41)
**Testo**: It is not suggested to use the VL set to select among different models

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Validation Set è lo strumento fondamentale raccomandato per selezionare i modelli e gli iperparametri.

---

### 📌 Domanda 42 (ID: 42)
**Testo**: It is a good practice to stop selecting the number of epochs that provides very low training error

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Fermare l'addestramento scegliendo le epoche solo in base all'errore di training minimo conduce quasi certamente all'Overfitting.

---

### 📌 Domanda 43 (ID: 43)
**Testo**: Both early stopping and regularization Tikhonov aims to control the effective complexity of the model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Sia l'Early Stopping (che limita il numero di epoche) sia la Regolarizzazione di Tikhonov $L_2$ (che limita la norma $\|w\|^2$) controllano la complessità effettiva (capacità VC) del modello.

---

### 📌 Domanda 44 (ID: 44)
**Testo**: The momentum in the batch and on-line version has the same meaning and effect

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Momentum in Batch considera il gradiente dell'intero dataset/batch precedente, mentre in On-Line considera il gradiente del singolo pattern precedente.

---

### 📌 Domanda 45 (ID: 45)
**Testo**: The validation and training sets compose the “design set”

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Training set e il Validation set insieme formano il cosiddetto 'Design Set' / 'Development Set', usato per costruire e selezionare il modello finale.

---

### 📌 Domanda 46 (ID: 46)
**Testo**: The K-fold CV is an approach to split the dataset

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La K-Fold Cross-Validation è uno schema di partizionamento dei dati in $K$ blocchi per valutare le prestazioni senza sprecare dati.

---

### 📌 Domanda 47 (ID: 47)
**Testo**: Selecting the number of units of a NN by the grid search is enough, we can avoid other regularization approaches,True False

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Grid Search sul numero di unità regola solo la struttura, ma servono altre forme di regolarizzazione (es. Weight Decay, Dropout, Early Stopping) per prevenire l'overfitting.

---

### 📌 Domanda 48 (ID: 48)
**Testo**: During the training the complexity of the NN increases

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Man mano che l'addestramento prosegue nelle epoche, i pesi crescono di norma e la funzione appresa diventa più complessa, aumentando la capacità effettiva della rete.

---

### 📌 Domanda 49 (ID: 49)
**Testo**: The model selection does not consider the TS error

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Test Set deve rimanere sigillato durante la Model Selection per evitare la contaminazione delle informazioni.

---

### 📌 Domanda 50 (ID: 50)
**Testo**: The training error is the best indicator to choose among different models

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'errore di training diminuisce sempre all'aumentare della complessità e non indica la capacità di generalizzazione; l'indicatore corretto è l'errore di validazione.

---

### 📌 Domanda 51 (ID: 51)
**Testo**: The test error is the best indicator to choose among different models

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'errore di test serve per il Model Assessment finale, non per scegliere i modelli durante la Model Selection.

---

### 📌 Domanda 52 (ID: 52)
**Testo**: The lambda and the patient are hyperparameters

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> I parametri $\lambda$ (Weight Decay) e $patience$ (Early Stopping) sono iperparametri definiti esternamente dall'utente.

---

### 📌 Domanda 53 (ID: 53)
**Testo**: R-Prop is a variant of the backpropagation based algorithm in which the exact value of the gradient does not care

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> R-Prop (Resilient Backpropagation) utilizza solo il SEGNO del gradiente (positivo/negativo) per aggiornare i pesi, ignorando la sua magnitudo assoluta che potrebbe svanire.

---

### 📌 Domanda 54 (ID: 54)
**Testo**: It is suggested to use the VL set to estimate the general risk R of the SLT for assessment purposes

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Per valutare il rischio generale $R$ della SLT ai fini dell'assessment finale si usa il Test Set, non il Validation Set.

---

### 📌 Domanda 55 (ID: 55)
**Testo**: The validation set is used to estimate the $$R_{emp}$$ as used in the SLT

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'errore empirico $R_{emp}$ nel bound della SLT viene stimato sul dataset di addestramento/test, non sul Validation set.

---

### 📌 Domanda 56 (ID: 56)
**Testo**: Too many units can lead to overfitting unless proper regulations is adopted

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un numero eccessivo di unità nascoste aumenta la dimensione VC; senza opportuna regolarizzazione causa Overfitting.

---

### 📌 Domanda 57 (ID: 57)
**Testo**: H shatters X if and only if for every possible dichotomy of X, there exists a h in H that classifies perfectly all the data

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un insieme di punti $X$ è shatterato (frammentato) da $\mathcal{H}$ se $\mathcal{H}$ può realizzare tutte le $2^{|X|}$ possibili combinazioni di etichette con 0 errori.

---

### 📌 Domanda 58 (ID: 58)
**Testo**: The VC dimension of a class of functions H is the maximum cardinality of a set (configuration) of points in X that can discriminated by a model in H for at least a labeling assignment

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La VC-Dimension è la MASSIMA cardinalità $P$ per cui esiste ALMENO UN insieme di $P$ punti shatterabile (cioè che realizza TUTTE le $2^P$ dicotomie, non solo una assegnazione di etichette).

---

### 📌 Domanda 59 (ID: 59)
**Testo**: The objective function of the primal form of the SVM hard margin minimizes the training errors

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'obiettivo del problema primale della Hard Margin SVM è minimizzare $\frac{1}{2}\|w\|^2$ (massimizzare il margine), assumendo 0 errori di addestramento (vincoli rigidi $y_i(w^T x_i+b) \ge 1$).

---

### 📌 Domanda 60 (ID: 60)
**Testo**: In the SVM hard-margin the minimization of the norm of w leads to the increment of the $$R_{emp}$$ of the SLT

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Minimizzare la norma $\|w\|^2$ equivale a MASSIMIZZARE il margine $M = 2/\|w\|$, il che RIDUCE la dimensione VC e la capacità del modello, riducendo (migliorando) il termine di confidenza VC della SLT.

---

### 📌 Domanda 61 (ID: 61)
**Testo**: The theorem of Vapnik shows for a SVM hard margin the relationships between the VC-dim and the margin

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il teorema di Vapnik per le SVM dimostra la relazione inversa tra l'ampiezza del margine $M = 2/\|w\|$ e la VC-dimension: $VC \le \min(D, R^2/\gamma^2) + 1$.

---

### 📌 Domanda 62 (ID: 62)
**Testo**: For a ML model with a n-dimensional input space, the VC-dim is n + 1

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La VC-dimension della classe degli iperpiani lineari (LTU) in uno spazio a $n$ dimensioni $\mathbb{R}^n$ è esattamente $n + 1$ (dimostrato via Teorema di Radon).

---

### 📌 Domanda 63 (ID: 63)
**Testo**: A support vector is a positive value

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un Vettore di Supporto non è uno scalare positivo, ma un vettore d'ingresso $x^{(s)} \in \mathbb{R}^D$ che giace esattamente sul bordo del margine $y^{(s)}(w^T x^{(s)} + b) = 1$.

---

### 📌 Domanda 64 (ID: 64)
**Testo**: The SVM hard margin (under its assumption of use) does not need any hyperparameters

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Sotto l'assunzione di separabilità lineare, la Hard Margin SVM non richiede iperparametri (a differenza della Soft Margin che richiede $C$).

---

### 📌 Domanda 65 (ID: 65)
**Testo**: Some configuration of 3 points in the plane are not shattered by a linear model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Tre punti disposti in linea retta (collineari) nel piano $\mathbb{R}^2$ non possono essere shatterati da un iperpiano lineare se le etichette alternate sono $+1, -1, +1$.

---

### 📌 Domanda 66 (ID: 66)
**Testo**: An optimal hyperplane is the one with minimum margin

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'iperpiano ottimale nelle SVM è quello che MASSIMIZZA il margine di separazione ($M = 2/\|w\|$), non quello con margine minimo.

---

### 📌 Domanda 67 (ID: 67)
**Testo**: The objective function of the SVM hard margin is related to the control of the VC-dim of the model

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Massimizzare il margine nelle SVM riduce la norma pesi e controlla direttamente la dimensione VC dello spazio delle ipotesi.

---

### 📌 Domanda 68 (ID: 68)
**Testo**: Bagging for classification is likely to decrease the margin of the classifier

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Bagging nella classificazione aumenta (non diminuisce) il margine medio del classificatore grazie all'effetto di mediazione dei vettori di decisione.

---

### 📌 Domanda 69 (ID: 69)
**Testo**: Bagging can exploit high variance learners

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Bagging funziona particolarmente bene con modelli ad ALTA VARIANZA (es. alberi di decisione profondi o reti non regolarizzate), riducendo la varianza mediante averaging.

---

### 📌 Domanda 70 (ID: 70)
**Testo**: Boosting cannot be used with weak learners

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Boosting è progettato specificamente per combinare progressivamente una sequenza di modelli deboli (weak learners) trasformandoli in un modello forte.

---

### 📌 Domanda 71 (ID: 71)
**Testo**: Boosting differentiates each training progressively concentrating on well classified patterns

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Boosting ripesa il dataset concentrandosi progressivamente sui pattern ERRATI (misclassified) e non su quelli ben classificati.

---

### 📌 Domanda 72 (ID: 72)
**Testo**: A deep neural network allows for multiple levels of abstraction of the internal representations

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Le reti neurali profonde apprendono gerarchie di rappresentazione distribuiti a livelli di astrazione via via crescenti ad ogni strato.

---

### 📌 Domanda 73 (ID: 73)
**Testo**: A deep (i.e. not shallow) neural network is the only way to implement a representation learning

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Sebbene le NNs profonde siano molto efficaci, esistono altri metodi per il representation learning (es. Autoencoders intelligenti, Kernel metodi, SOM, RBF networks).

---

### 📌 Domanda 74 (ID: 74)
**Testo**: With a two layer logic network is possible to solve the parity problem with polynomial number of gates with respect to the input dimension

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Per risolvere il problema della parità a $N$ bit con una rete logica a soli 2 strati occorre un numero ESPONENZIALE di porte logiche ($2^{N-1}+1$).

---

### 📌 Domanda 75 (ID: 75)
**Testo**: An autoencoder can be used to initialize a deep NN exploiting the decoder (pre)trained part

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Quando si pre-addestra un Autoencoder per inizializzare una rete profonda, si conserva la parte ENCODER $W_1$ e si scarta la parte Decoder $W'$.

---

### 📌 Domanda 76 (ID: 76)
**Testo**: Stacking many layers with sigmoidal functions and using backprogation, gradient vanish can occur

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Moltiplicando le derivate della Sigmoide ($\le 0.25$) lungo molti strati durante la Backpropagation, il gradiente tende a zero (Gradient Vanishing).

---

### 📌 Domanda 77 (ID: 77)
**Testo**: Gradient clipping check the norm of the gradient to bound the weights update

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Gradient Clipping controlla la norma del gradiente $\|g\|$: se $\|g\| > v$, ridimensiona il gradiente $g \leftarrow v \frac{g}{\|g\|}$ per evitare esplosioni nell'aggiornamento pesi.

---

### 📌 Domanda 78 (ID: 78)
**Testo**: Increasing the number of layers in a logic network is possible to solve the parity problem with polynomial number of gates with respect to the input dimension

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Aumentando il numero di strati a $\log N$ in una rete logica ad albero binario, il problema della parità si risolve con un numero POLINOMIALE di porte $3(N-1)$.

---

### 📌 Domanda 79 (ID: 79)
**Testo**: Choose the appropriate claims (with, as usual, negative score for the not correct options):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 2, Opzione 5, Opzione 7**

**Opzioni:**
  1. [✅] “No flattening” results show that for some functions, a shallow model would require an exponential number of units with respect to the dimension of the input
  2. [✅] “No flattening” results show that a deep model can be more efficient than a shallow one, with an exponential gain in terms of units
  3. [❌] “No flattening” results show that a deep model can be more efficient than a shallow one, with an exponential gain in terms of bits used to encode each weight of the network
  4. [❌] “No flattening” results show that a deep model is more expressive than a shallow one, with an exponential gain in terms of functions that can be represented
  5. [✅] “No flattening” results does not show that a deep model is more expressive than a shallow one, but that we can have exponential gain in terms of number of weights
  6. [❌] “No flattening” results show that for some functions, a shallow model would require an exponential number of units with respect to the number of layers
  7. [✅] “No flattening” results show that some compositional functions cannot be implemented by a shallow network holding the same number of units of a deep network
  8. [❌] “No flattening” results show that a deep models cannot be more efficient, in terms of units, than a shallow one

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> I risultati di 'No Flattening' dimostrano che le architetture profonde possono rappresentare in modo compatto (con un guadagno esponenziale sul numero di unità e pesi) funzioni che in architetture shallow richiederebbero un numero esponenziale di neuroni.

---

### 📌 Domanda 80 (ID: 80)
**Testo**: Choose from the following one or more alternatives of formulations for the bias/variance decomposition that are correct (with the notation and assumption used in the lecture of the course). All appropriate answers should be listed below (to gain the max score), but those indicated as appropriate that are not, however, affect the exercise score by negative values

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 3**

**Opzioni:**
  1. [❌] $E_P[(y - h(x))^2] = (f(x) - \bar{h}(x))^2 + E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible data points x according to a distribution P.
  2. [✅] $E_P[(y - h(x))^2] = E_P[(h(x) - \bar{h}(x))^2] + (f(x) - \bar{h}(x))^2 + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  3. [✅] $E_P[(y - h(x))^2] = (f(x) - E_P[h(x)])^2 + E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  4. [❌] $E_P[(y - h(x))^2] = E_P[(h(x) - \bar{h}(x))^2] + (f(x) - h(x))^2 + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  5. [❌] $E_P[(y - h(x))^2] = (f(x) - \bar{h}(x))^2 - E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible training sets sampled according to a distribution P.
  6. [❌] $E_P[(y - h(x))^2] = (f(x) - h(x))^2 + E_P[(h(x) - \bar{h}(x))^2] + \sigma^2$ with the mean computed with respect to all the possible data points x according to a distribution P.

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La formula corretta della scomposizione Bias-Varianza calcola il valore atteso rispetto a tutti i possibili TRAINING SET estratti dalla distribuzione $P$: $\mathbb{E}_P[(y-h(x))^2] = \text{Bias}^2 + \text{Var} + \sigma^2$.

---

### 📌 Domanda 81 (ID: 81)
**Testo**: Choose one or more alternatives of formulations of the RBF kernel that are correct. All appropriate answers should be listed below (to gain the max score), but those indicated as appropriate that are not, however, affect the exercise by a negative score.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 5, Opzione 7, Opzione 8**

**Opzioni:**
  1. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2}\|v-t\|^2}$
  2. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2}(t-v)}$
  3. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2}\|t-v\|}$
  4. [❌] $K(t,v) = - e^{\frac{1}{2\sigma^2}\|t-v\|^2}$
  5. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2}\|v-t\|^2}$
  6. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2}\|t-v\|}$
  7. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2}\|t-v\|^2}$
  8. [✅] $K(v,t) = e^{-\frac{1}{2\sigma^2}\|v-t\|^2}$
  9. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2}\|t-v\|^2}$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Le formulazioni corrette del Kernel RBF Gaussiano richiedono l'esponente negativo $-\frac{1}{2\sigma^2}$ e la norma euclidea al quadrato $\|t-v\|^2$. Data la simmetria, $K(t,v) = K(v,t)$.

---

### 📌 Domanda 82 (ID: 82)
**Testo**: List the proper claims for the Soft margin SVM

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 5, Opzione 6, Opzione 7**

**Opzioni:**
  1. [❌] Small values of C can lead to overfitting
  2. [✅] Slack variables change the constraints of the primal problem
  3. [❌] The slack variables have no effect on the width of the margin
  4. [❌] High values of C can lead to underfitting
  5. [✅] With slack variables is possible to gain a greater margin
  6. [✅] The values of \alpha allow us to know the not-null slack variables
  7. [✅] C in a soft margin SVM can be tailored by the user with a cross validation approach
  8. [❌] Slack variables does not changes the constraints of the primal problem

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nella Soft Margin SVM: $C$ piccolo tollera più violazioni ed evita l'overfitting; le slack variables $\xi_i$ modificano i vincoli primali in $y_i(w^T x_i+b) \ge 1-\xi_i$ ed ampliano il margine.

---

### 📌 Domanda 83 (ID: 83)
**Testo**: Compute the correct delta for a hidden unit u or z that is connected in output to units denoted by the index v, using the conventions assumed in the course.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 4, Opzione 6**

**Opzioni:**
  1. [❌] $\delta_u = (\sum_u \delta_v w_{vu}) f'_u(net_u)$
  2. [❌] $\delta_z = (\sum_z \delta_v w_{vz}) f'_z(net_z)$
  3. [❌] $\delta_u = (\sum_u \delta_u w_{vu}) f'_u(net_u)$
  4. [✅] $\delta_u = (\sum_v \delta_v w_{vu}) f'_u(net_u)$
  5. [❌] $\delta_z = (\sum_v \delta_v w_{zv}) f'_z(net_z)$
  6. [✅] $\delta_z = (\sum_v \delta_v w_{vz}) f'_z(net_z)$
  7. [❌] $\delta_z = (\sum_z \delta_z w_{vz}) f'_z(net_z)$
  8. [❌] $\delta_u = (\sum_v \delta_v w_{uv}) f'_u(net_u)$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il fattore di errore locale per un neurone nascosto $u$ o $z$ connesso ai neuroni successivi $v$ è $\delta_u = (\sum_v \delta_v w_{vu}) f'_u(net_u)$.

---

### 📌 Domanda 84 (ID: 84)
**Testo**: Compute the correct delta for a hidden unit r or s that is connected in output to units denoted by the index v, using the conventions assumed in the course.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3, Opzione 5**

**Opzioni:**
  1. [❌] $\delta_s = (\sum_s \delta_s w_{vs}) f'_s(net_s)$
  2. [❌] $\delta_s = (\sum_v \delta_s w_{uv}) f'_s(net_s)$
  3. [✅] $\delta_s = (\sum_v \delta_v w_{vs}) f'_s(net_s)$
  4. [❌] $\delta_r = (\sum_r \delta_v w_{vr}) f'_r(net_r)$
  5. [✅] $\delta_r = (\sum_v \delta_v w_{vr}) f'_r(net_r)$
  6. [❌] $\delta_s = (\sum_s \delta_v w_{vs}) f'_s(net_s)$
  7. [❌] $\delta_r = (\sum_r \delta_r w_{vr}) f'_r(net_r)$
  8. [❌] $\delta_r = (\sum_v \delta_v w_{rv}) f'_r(net_r)$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La formula del $\delta$ per un neurone nascosto $s$ o $r$ collegato ai neuroni di output $v$ è $\delta_s = (\sum_v \delta_v w_{vs}) f'_s(net_s)$.

---

### 📌 Domanda 85 (ID: 85)
**Testo**: Given an unit with activation function f and δ_p = (d_p - f(net(x_p))), and denoting the weights with t or i, list the proper results among the following equations for a pattern p:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3, Opzione 6**

**Opzioni:**
  1. [❌] $\frac{\partial E_p(w)}{\partial w_t} = +2\delta_p (x_{p,t}) f'(x_p) net(x_p)$
  2. [❌] $\frac{\partial E_p(w)}{\partial w_i} = -2\delta_p (x_{p,i}) f'(x_p)$
  3. [✅] $\frac{\partial E_p(w)}{\partial w_t} = -2 x_{p,t} \delta_p f'(net(x_p))$
  4. [❌] $\frac{\partial E_p(w)}{\partial w_i} = +2 \delta_p f'(net(x_p)) x_{p,i}$
  5. [❌] $\frac{\partial E_p(w)}{\partial w_t} = -2 x_{p,t} f'(net(x_p)) net(x_p)$
  6. [✅] $\frac{\partial E_p(w)}{\partial w_i} = -2 \delta_p f'(net(x_p)) x_{p,i}$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La derivata parziale per un pattern $p$ è $\frac{\partial E_p}{\partial w_i} = -2 \delta_p f'(net(x_p)) x_{p,i}$, in virtù della Chain Rule.

---

### 📌 Domanda 86 (ID: 86)
**Testo**: According to the picture and notation in the slide “Exercise to use a uniform notation for the inputs” of Lecture on NN-part1, what is the right expression for the net of a hidden unit? (note that the input argument of the net has been omitted)

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 6**

**Opzioni:**
  1. [❌] $net_j = \sum_j w_{ji} o_i$
  2. [❌] $net_i = \sum_i w_{ij} o_i$
  3. [❌] $net_k = \sum_j w_{kj} o_j$
  4. [❌] $net_j = \sum_j w_{ij} o_j$
  5. [❌] $net_j = \sum_i w_{ij} o_i$
  6. [✅] $net_j = \sum_i w_{ji} o_i$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Per la notazione del corso (input $i$, hidden $j$, output $k$), l'ingresso netto al neurone nascosto $j$ è $net_j = \sum_i w_{ji} o_i$.

---

### 📌 Domanda 87 (ID: 87)
**Testo**: List the proper claims for the K-NN:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 3, Opzione 5, Opzione 7**

**Opzioni:**
  1. [✅] Neighborhoods are no longer "local" when the input dimension n increases
  2. [❌] The curse of dimensionality is due to the high retrieval cost for testing data in K-NN
  3. [✅] The curse of dimensionality is due to the changing in the data density when we change the input dimension
  4. [❌] A high value of K leads to very flexible models
  5. [✅] Lower values of K, for the K-NN, lead to more flexible models
  6. [❌] For the K-NN: if the dimension of input increases, we can compensate the curse of dimensionality reducing the value of K so to avoid negative effects on the generalization capability
  7. [✅] For the K-NN, fixing K: with high dimensional inputs, the range of features values to be considered becomes high, so you lose the similarity among examples in the N(x)

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nei k-NN ad alta dimensione, la densità dei dati diminuisce drammaticamente (curse of dimensionality), i vicinati perdono la loro natura 'locale' e il concetto di similarità sfuma.

---

### 📌 Domanda 88 (ID: 88)
**Testo**: The use of a Tikhonov loss introduces a new learning rule where we have a further addend for the gradient of MSE given by:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 6**

**Opzioni:**
  1. [❌] $+ 2 \lambda w$
  2. [❌] $- 2 \lambda w$
  3. [❌] $- 2 \lambda$
  4. [❌] $+ 2 \lambda$
  5. [❌] $+ 2 \lambda w$
  6. [✅] $- 2 \lambda w$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Aggiungendo la regolarizzazione di Tikhonov $\lambda \|w\|^2$ alla loss, la derivata del termine di regolarizzazione porta l'addendo $-2\lambda w$ nell'aggiornamento dei pesi per discesa del gradiente.

---

### 📌 Domanda 89 (ID: 89)
**Testo**: A LTU for a pattern p can be expressed as (assuming x0 = 1):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 4, Opzione 5, Opzione 8**

**Opzioni:**
  1. [✅] $h(x_p) = sign(x_p^T w)$
  2. [❌] $h(x_p) = sign(x_p^T w_p)$
  3. [❌] $h(x_p) = sign(\sum_{t=0}^n x_{p,t} w_u)$
  4. [✅] $h(x_p) = sign(\sum_{t=0}^n w_t x_{p,t})$
  5. [✅] $h(x_p) = sign(\sum_{u=0}^n x_{p,u} w_u)$
  6. [❌] $h(x_p) = sign(w_p^T x_p)$
  7. [❌] $h(x_p) = sign(\sum_{u=0}^n x_{p,u} w_t)$
  8. [✅] $h(x_p) = sign(w^T x_p)$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'output di una LTU con bias incorporato $x_0=1$ è $h(x_p) = \text{sign}(w^T x_p) = \text{sign}(x_p^T w) = \text{sign}(\sum_{u=0}^n w_u x_{p,u})$.

---

### 📌 Domanda 90 (ID: 90)
**Testo**: Considering the SLT, fixing l and δ (delta), a polynomial with increased/higher degree:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 4, Opzione 5, Opzione 6**

**Opzioni:**
  1. [❌] It implies higher R value
  2. [❌] It implies lower R value
  3. [❌] It implies that the VC-bound does not hold
  4. [✅] It has higher VC-confidence
  5. [✅] It has higher VC-dim
  6. [✅] It provides higher fitting capability, reducing $R_{emp}$
  7. [❌] It provides higher fitting capability, increasing $R_{emp}$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Aumentando il grado del polinomio, si aumentano la capacità del modello, la VC-dimension e la VC-confidence, riducendo l'errore empirico $R_{emp}$.

---

### 📌 Domanda 91 (ID: 91)
**Testo**: Considering the SLT, fixing l and δ (delta), a polynomial with decreased/lower degree:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 3**

**Opzioni:**
  1. [❌] It implies higher R value
  2. [✅] It provides lower fitting capability, increasing R_{emp}
  3. [✅] It has lower VC-dim
  4. [❌] It implies lower R value
  5. [❌] It implies that the VC-bound does not hold

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Riducendo il grado del polinomio, la VC-dimension diminuisce ed il modello ha minore capacità di fitting, aumentando l'errore empirico $R_{emp}$.

---

### 📌 Domanda 92 (ID: 92)
**Testo**: List all the appropriate answers (but those indicated as appropriate that are not affect the exercise sum by a negative score).

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 4, Opzione 5, Opzione 8**

**Opzioni:**
  1. [✅] In a regression task each data is labeled with a real number
  2. [❌] In a classification task we search for subsets of "similar" data
  3. [❌] The hypothesis space contains all the input data
  4. [✅] In supervised learning for classification the labeled examples can be provided in the form <x,+1> e <x,-1>
  5. [✅] In the supervised learning the aim is to approximate an unknown target function
  6. [❌] In a unsupervised learning the labeled examples are provided in the form <x,d>
  7. [❌] In a multi-class problem f can have K continuous values in output
  8. [✅] If we have a Boolean target function we learn a classifier with two classes

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nella regressione il target è reale; nella classificazione binaria le etichette possono essere $\langle x, +1\rangle$ e $\langle x, -1\rangle$; l'obiettivo supervisionato è approssimare la funzione target $f$.

---

### 📌 Domanda 93 (ID: 93)
**Testo**: The negative of this vector provides the direction toward local minima of a function f(x1, x2) in a 2-dimensional space: (∂f/∂x1, ∂f/∂x2)

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il vettore gradiente $\nabla f = (\partial f/\partial x_1, \partial f/\partial x_2)$ punta nella direzione di massima crescita; il suo opposto $-\nabla f$ punta verso i minimi locali.

---

### 📌 Domanda 94 (ID: 94)
**Testo**: List all the appropriate answers (but those indicated as appropriate that are not affect the exercise sum by a negative score).

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 4**

**Opzioni:**
  1. [✅] Changing the nature (type) of the target we can see classification or regression as a function approximation task.
  2. [❌] In a unsupervised learning the labeled examples are provided in the form <x,d>.
  3. [❌] In a classification task we search for subsets of 'similar' data.
  4. [✅] If we have a Boolean target function we learn a classifier with two classes.

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Variando la natura del target (discreto per classificazione, continuo per regressione), entrambi i compiti sono formalmente inquadrati come approssimazione di funzioni.

---

### 📌 Domanda 95 (ID: 95)
**Testo**: There is no inductive bias called "full space bias".

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Non esiste alcun bias induttivo formalmente denominato 'full space bias'.

---

### 📌 Domanda 96 (ID: 96)
**Testo**: There exists the version space inductive bias.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Version Space Inductive Bias assume che la funzione target appartenga al Version Space (l'insieme delle ipotesi consistenti con i dati di train).

---

### 📌 Domanda 97 (ID: 97)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the model response and the outputs mean.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La loss function nell'apprendimento supervisionato misura lo scarto tra la predizione del modello e il target reale, non la media delle uscite.

---

### 📌 Domanda 98 (ID: 98)
**Testo**: The VC dimension of a class of functions H is the maximum cardinality of a set (configuration) of points in X that can be discriminated by a model in H for at least a labeling assignment.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La VC-dimension richiede la capacità di shatterare l'insieme di punti (realizzando TUTTE le $2^P$ combinazioni), non solo almeno un'assegnazione.

---

### 📌 Domanda 99 (ID: 99)
**Testo**: The VC-dim can be used to define a nested hierarchy over hypothesis spaces of growing complexity.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La VC-dimension definisce la capacità dello spazio delle ipotesi, permettendo di costruire la gerarchia annidata $\mathcal{H}_1 \subset \mathcal{H}_2 \dots$ del principio SRM.

---

### 📌 Domanda 100 (ID: 100)
**Testo**: The objective function of the primal form of the SVM hard margin minimizes the training errors.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il problema primale della Hard Margin SVM minimizza $\frac{1}{2}\|w\|^2$ (massimizzando il margine) assumendo 0 errori di addestramento.

---

### 📌 Domanda 101 (ID: 101)
**Testo**: Choose one or more alternatives of formulations of the RBF kernel that are correct. All appropriate answers should be listed below (to gain the max score), but those indicated as appropriate that are not, however, affect the exercise by a negative score.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 7, Opzione 8**

**Opzioni:**
  1. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2} \|t-v\|^2}$
  2. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2} (t-v)}$
  3. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2} \|t-v\|^2}$
  4. [❌] $K(t,v) = e^{-\frac{1}{2\sigma^2} \|t-v\|}$
  5. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2} \|t-v\|}$
  6. [❌] $K(t,v) = -e^{-\frac{1}{2\sigma^2} \|t-v\|^2}$
  7. [✅] $K(v,t) = e^{-\frac{1}{2\sigma^2} \|v-t\|^2}$
  8. [✅] $K(t,v) = e^{-\frac{1}{2\sigma^2} \|v-t\|^2}$
  9. [❌] $K(t,v) = e^{\frac{1}{2\sigma^2} \|v-t\|^2}$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Le espressioni valide del Kernel RBF Gaussiano richiedono l'esponente negativo $-\frac{1}{2\sigma^2}$ e la norma euclidea al quadrato $\|t-v\|^2$.

---

### 📌 Domanda 102 (ID: 102)
**Testo**: The cascade correlation algorithm allows to automatically find the number of hidden units.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'algoritmo Cascade Correlation aggiunge progressivamente neuroni nascosti, determinando automaticamente la dimensione ideale dell'architettura.

---

### 📌 Domanda 103 (ID: 103)
**Testo**: Both early stopping and regularization Tikhonov aims to control the effective complexity of the model.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Sia l'Early Stopping che la regolarizzazione di Tikhonov controllano la norma pesi e la capacità effettiva del modello per prevenire l'overfitting.

---

### 📌 Domanda 104 (ID: 104)
**Testo**: The momentum in the batch and on-line version has the same meaning and effect.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Momentum in Batch opera sul gradiente medio del batch precedente, mentre in On-Line opera sul gradiente del singolo esempio precedente.

---

### 📌 Domanda 105 (ID: 105)
**Testo**: The K-fold CV is an approach to split the dataset.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La K-Fold Cross-Validation suddivide il dataset in $K$ fold ruotando il blocco di validazione.

---

### 📌 Domanda 106 (ID: 106)
**Testo**: The model selection does not consider the TR error.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Model Selection valuta gli iperparametri sull'errore di validazione, ignorando l'errore di training (TR error) che è polarizzato.

---

### 📌 Domanda 107 (ID: 107)
**Testo**: A NN with 1 hidden layer (using logistic activation functions) has a universal approximation capability.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Teorema di Approssimazione Universale (Hornik 1989) garantisce che una rete MLP con 1 strato nascosto e attivazione sigmoidale può approssimare qualsiasi funzione continua.

---

### 📌 Domanda 108 (ID: 108)
**Testo**: In a NN by using multiple output units we can obtain a multi-regression or multi-classes classifier.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Utilizzando più unità di output si possono realizzare modelli di regressione multivariata (es. MEE 4D per la CUP) o classificatori multi-classe.

---

### 📌 Domanda 109 (ID: 109)
**Testo**: ∂E/∂w = (∂E/∂o) · (∂w/∂o)

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Per la Chain Rule, $\frac{\partial E}{\partial w} = \frac{\partial E}{\partial o} \cdot \frac{\partial o}{\partial w}$, non il reciproco $\frac{\partial w}{\partial o}$.

---

### 📌 Domanda 110 (ID: 110)
**Testo**: The backpropagation allows us to obtain the update rule for each weight in the network on the basis of an exact gradient computing.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La Backpropagation calcola il gradiente esatto della funzione di costo rispetto a ciascun peso applicando la regola della catena.

---

### 📌 Domanda 111 (ID: 111)
**Testo**: In the backpropagation the delta values are propagated from the input to the output layers units.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nella Backpropagation i segnali d'errore $\delta$ vengono retropropagati dall'output verso gli strati nascosti e l'input (da destra a sinistra).

---

### 📌 Domanda 112 (ID: 112)
**Testo**: According to the Backpropagation, given a TR set with l patterns, the $$ \Delta w $$ can be expressed as (list the proper results among the following equations):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 4, Opzione 6**

**Opzioni:**
  1. [❌] $\Delta w_{rh} = \sum_{p=1}^{l} \delta_{p,r} \; o_{p,h}$
  2. [❌] $\Delta w_{ts} = - \sum_{p=1}^{l} \delta_{p,t} \; o_{p,s}$
  3. [❌] $\Delta w_{rh} = \sum_{p=1}^{l} \delta_{p,h} \; o_{p,r}$
  4. [✅] $\Delta w_{rh} = - \sum_{p=1}^{l} \delta_{p,r} \; o_{p,h}$
  5. [❌] $\Delta w_{ts} = - \sum_{p=1}^{l} \delta_{p,s} \; o_{p,t}$
  6. [✅] $\Delta w_{ts} = \sum_{p=1}^{l} \delta_{p,t} \; o_{p,s}$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'aggiornamento pesi sull'intero dataset è proporzionale alla somma dei prodotti tra il $\delta$ del neurone ricevente e l'output $o$ del neurone mittente: $\Delta w_{ts} = \sum_{p=1}^l \delta_{p,t} o_{p,s}$.

---

### 📌 Domanda 113 (ID: 113)
**Testo**: The w of a NN are hyper-parameters to be selected by the user.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> I pesi $w$ di una rete neurale sono parametri interni ottimizzati durante l'addestramento, non iperparametri impostati dall'utente.

---

### 📌 Domanda 114 (ID: 114)
**Testo**: Mini-batch is a technique introduced to optimize the complexity of the model.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il Mini-batch è una tecnica di ottimizzazione stocastica del gradiente, non un metodo per modificare la capacità intrinseca del modello.

---

### 📌 Domanda 115 (ID: 115)
**Testo**: It is always convenient to have a high eta value to speed-up the training.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un tasso di apprendimento $\eta$ eccessivo causa oscillazioni instabili ed impedisce la convergenza al minimo della loss.

---

### 📌 Domanda 116 (ID: 116)
**Testo**: $w_{qr}$ denote the weight of the unit r coming from the input with index q.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nella notazione del corso, $w_{rq}$ indica il peso del neurone ricevente $r$ proveniente dall'input $q$ (il primo indice è il ricevente).

---

### 📌 Domanda 117 (ID: 117)
**Testo**: A LTU for a pattern p can be expressed as (assuming x0 = 1):

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 2, Opzione 3**

**Opzioni:**
  1. [❌] $h(x_p) = sign(\sum_{t=0}^{n} x_{p,t} w_u)$
  2. [✅] $h(x_p) = sign(x_p^T w)$
  3. [✅] $h(x_p) = sign(\sum_{u=0}^{n} x_{p,u} w_u)$
  4. [❌] $h(x_p) = sign(w_p^T x_p)$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'output di un LTU per il pattern $p$ è $h(x_p) = \text{sign}(x_p^T w) = \text{sign}(\sum_{u=0}^n w_u x_{p,u})$.

---

### 📌 Domanda 118 (ID: 118)
**Testo**: A linear classifier with LSE cannot solve a non-linear separable problem.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Un classificatore lineare combinato con Linear Basis Expansion (LBE) può separare problemi non linearmente separabili nello spazio di partenza.

---

### 📌 Domanda 119 (ID: 119)
**Testo**: List the proper claims for the Tikhonov regularization.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3, Opzione 4**

**Opzioni:**
  1. [❌] The regularization term does not depend on the size of w
  2. [❌] High lambda value allows maximum fitting of the training set
  3. [✅] The best lambda value allows us to optimize the model complexity
  4. [✅] Lambda = 0 corresponds to a minimum regularization effect

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La regolarizzazione di Tikhonov $E(w) + \lambda \|w\|^2$ bilancia il fitting col controllo della norma; $\lambda=0$ elimina la regolarizzazione.

---

### 📌 Domanda 120 (ID: 120)
**Testo**: List the proper claims for the K-NN.

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 1, Opzione 2**

**Opzioni:**
  1. [✅] For the K-NN, fixing K with high dimensional inputs, the range of features values to be considered becomes high, so you lose the similarity among examples in the N(x)
  2. [✅] The curse of dimensionality is due to the changing in the data density when we change the input dimension
  3. [❌] A high value of K leads to a very flexible models
  4. [❌] If the dimension of input increases, we can compensate the curse of dimensionality reducing the value of K so to avoid negative effects on the generalization capability

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nei k-NN in spazi ad alta dimensione, la densità dei dati crolla (curse of dimensionality) e la nozione di vicinato locale perde di significato.

---

### 📌 Domanda 121 (ID: 121)
**Testo**: $||x||_2 = \sum_i x_i^2$

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La norma $L_2$ è $\|x\|_2 = \sqrt{\sum x_i^2}$. La quantità $\sum x_i^2$ rappresenta la norma al quadrato $\|x\|_2^2$.

---

### 📌 Domanda 122 (ID: 122)
**Testo**: The gradient of f can be written as a vector whose components are the partial derivatives of f.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Il vettore gradiente $\nabla f = (\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n})$ raccoglie tutte le derivate parziali prime della funzione.

---

### 📌 Domanda 123 (ID: 123)
**Testo**: In a regression task each data is labeled with a real number.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nei compiti di regressione ogni pattern è associato ad un etichetta target continua su valore reale $y \in \mathbb{R}$.

---

### 📌 Domanda 124 (ID: 124)
**Testo**: Removing the inductive bias from a learning system is positive (i.e., it brings advantages for the learning system).

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Senza bias induttivo, un algoritmo di apprendimento non ha alcun criterio razionale per classificare istanze mai viste prima.

---

### 📌 Domanda 125 (ID: 125)
**Testo**: For the supervised learning the loss is computed to evaluate the distance between the model response and its inputs.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> La loss function in apprendimento supervisionato valuta lo scarto tra la risposta predetta $h(x)$ e il target reale $y$.

---

### 📌 Domanda 126 (ID: 126)
**Testo**: The underfitting occurs because we have too few data.

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **FALSE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> L'underfitting è causato da un modello troppo semplice (bassa capacità), non dal numero di dati.

---

### 📌 Domanda 127 (ID: 127)
**Testo**: Given a unit with activation function f and $\delta_p = (d_p - f(net(x_p)))$, list the proper results among the following equations for a pattern p:

* **Tipo**: Scelta Multipla
* **Opzioni Corrette**: **Opzione 3**

**Opzioni:**
  1. [❌] $\frac{\partial E_p(w)}{\partial w_t} = -2\, x_{p,t}\, f'\!\big(net(x_p)\big) \, net(x_p)$
  2. [❌] $\frac{\partial E_p(w)}{\partial w_i} = -2\, \delta_p\, (x_{p,i})\, f'(x_p)$
  3. [✅] $\frac{\partial E_p(w)}{\partial w_i} = -2\, \delta_p \, f'\!\big(net(x_p)\big)\, x_{p,i}$
  4. [❌] $\frac{\partial E_p(w)}{\partial w_t} = -2\, x_{p,t}\, \delta_p \, f\!\big(net(x_p)\big)$
  5. [❌] $\frac{\partial E_p(w)}{\partial w_t} = +2\, \delta_p \, (x_{p,t}) \, f'(x_p)\, net(x_p)$
  6. [❌] $\frac{\partial E_p(w)}{\partial w_i} = +2\, \delta_p \, f'\!\big(net(x_p)\big)\, x_{p,i}$

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Per un pattern $p$, la derivata parziale rispetto al peso d'ingresso $i$ è $\frac{\partial E_p}{\partial w_i} = -2 \delta_p f'(net(x_p)) x_{p,i}$.

---

### 📌 Domanda 128 (ID: 128)
**Testo**: Given x input, d target, o the perceptron’s output, the update rule of the Perceptron Learning Algorithm: It is $w_new = w + η x d$ only if there is a classification error for x

* **Tipo**: Vero / Falso
* **Risposta Corretta**: **TRUE**

> [!NOTE]
> **Spiegazione & Derivazione Teorica**:
> Nell'algoritmo del Perceptrone, l'aggiornamento pesi $w_{new} = w + \eta x d$ viene eseguito ESCLUSIVAMENTE quando il pattern $x$ viene classificato in modo errato.

---
