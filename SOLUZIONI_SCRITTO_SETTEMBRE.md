# ✍️ Soluzioni Ufficiali ed Esaustive della Prova Scritta d'Esame
**Corso del Prof. Alessio Micheli — Università di Pisa**  
*(Compendio Integrale delle 4 Domande della Prova Scritta con Risposte Modello, Dimostrazioni Algebriche Passo-Passo, Diagrammi ASCII e Spiegazioni per l'Orale)*

---

## 📌 DOMANDA 1: Funzione di Attivazione ReLU e Caratteristiche Principali

### 1.1 Definizione Matematica, Subgradiente e Derivata Prima
La **ReLU (Rectified Linear Unit)** è definita matematicamente come:
$$f(x) = \max(0, x) = \begin{cases} x & \text{se } x > 0 \\ 0 & \text{se } x \le 0 \end{cases}$$

La sua derivata prima rispetto all'input è una funzione a gradino (interruttore binario):
$$f'(x) = \begin{cases} 1 & \text{se } x > 0 \\ 0 & \text{se } x < 0 \end{cases}$$

*(Nota per l'Orale: In $x=0$ la derivata non è definita in senso classico per la discontinuità dell'angolo. In analisi matematica si definisce il **subgradiente** $\partial f(0) = [0, 1]$. In ambito software/PyTorch si imposta convenzionalmente $f'(0) = 0$ oppure $f'(0) = 1$).*

#### Grafico ASCII dell'Attivazione f(x) e della Derivata f'(x)
```
     Funzione ReLU f(x)                  Derivata f'(x)
         y |                                 y |
           |   / (pendenza 1)                  |------- (valore 1)
           |  /                                |
   --------+--/----> x                 --------+--------> x
   (0 per x<0)                         (0 per x<0)
```

---

### 1.2 Caratteristiche Principali e Vantaggi

#### 1. Risoluzione del Vanishing Gradient per $x > 0$
Nelle attivazioni sigmoidali $\sigma(z) = \frac{1}{1+e^{-z}}$ o Tanh, la derivata satura per valori elevati di input ($f'(z) \le 0.25$ per la sigmoide).  
Durante la Backpropagation su una rete a $L$ strati, il gradiente rispetto ai pesi del primo strato è dato dalla regola della catena:
$$\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o_L} \left( \prod_{l=2}^L W_l f'_l(net_l) \right) f'_1(net_1) x$$
Poiché per le sigmoidali $f'_l \le 0.25$, la produttoria $\prod_{l=2}^L f'_l(net_l) \le (0.25)^{L-1}$ tende esponenzialmente a zero per $L \ge 4$, azzerando l'aggiornamento pesi (**Vanishing Gradient**).  
Per la ReLU, per tutti i neuroni attivi ($x > 0$), la derivata è **esattamente uguale a 1** ($f'(net) = 1$). Di conseguenza, la produttoria $\prod f'_l = 1$, consentendo al segnale d'errore di retropropagarsi in reti molto profonde senza alcuna attenuazione.

#### 2. Efficienza Computazionale
Non richiede il calcolo di funzioni euleriane o esponenziali complesse (come $e^{-z}$ in Sigmoide e Tanh), ma solo un'operazione di sogliatura elementare $\max(0, x)$ eseguibile a livello hardware in nanosecondi.

#### 3. Sparsità delle Attivazioni
Poiché per qualsiasi input negativo $x < 0$ l'output è esattamente $0$, per ogni pattern d'ingresso solo un sottoinsieme di neuroni della rete risulterà attivo ($f(x) > 0$). Questo produce rappresentazioni sparse che riducono il grado di accoppiamento dei pesi e migliorano la separabilità delle caratteristiche.

---

### 1.3 Svantaggi e Problema della "Dying ReLU"
* **Dying ReLU (Neuroni Morti)**:  
  Se un neurone riceve un gradiente elevato che spinge il suo vettore dei pesi a generare un input netto costantemente negativo ($net < 0$) per tutti i pattern del dataset, l'output sarà $0$ e la derivata sarà $0$. Di conseguenza, il segnale di errore $\delta = 0$ si azzera e il neurone "muore", rimanendo congelato senza potersi aggiornare mai più.
* **Non Zero-Centered**:  
  Gli output sono sempre non-negativi ($\ge 0$), inducendo una dinamica di aggiornamento dei pesi a zig-zag durante la discesa del gradiente.
* **Soluzioni Evolutive**:
  * **LeakyReLU**: $f(x) = \max(\alpha x, x)$ con $\alpha \approx 0.01$ per garantire una pendenza minima anche per $x < 0$.
  * **ELU (Exponential Linear Unit)**: $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$ per garantire una transizione fluida ed una media delle attivazioni vicina allo zero.
  * **GELU (Gaussian Error Linear Unit)**: $f(x) = x \cdot \Phi(x)$, adottata nei Transformer moderni.

---

## 📌 DOMANDA 2: Cosa è la Complessità nel Machine Learning

### 2.1 Definizione Generale
Nel Machine Learning, la **Complessità** misura la **capacità espressiva / ricchezza dello Spazio delle Ipotesi $\mathcal{H}$**, ovvero il grado di flessibilità con cui un modello è in grado di rappresentare confini di decisione non lineari o approssimare funzioni articolate a partire dai dati.

---

### 2.2 Le 3 Soluzioni Formali della Complessità nel Corso

#### 1. Complessità Strutturale / Capacità Misurata via Statistical Learning Theory (SLT)
* **VC-Dimension ($h_{VC}$)**: La massima cardinalità $N$ per cui esiste ALMENO UN insieme di $N$ punti shatterabile (es. $VC = D+1$ per iperpiani lineari in $\mathbb{R}^D$).
* **Bound di Vapnik & SRM**: Il termine di confidenza VC misura la penalità teorica di complessità:
  $$\Omega(N, h_{VC}) = \sqrt{\frac{h_{VC} \left( \ln(2N/h_{VC}) + 1 \right) - \ln(\eta/4)}{N}}$$
  Il principio di Structural Risk Minimization (SRM) seleziona l'ipotesi che bilancia Rischio Empirico $R_{emp}$ e Confidenza VC.

#### 2. Complessità Parametrica e di Regolarizzazione
* Misurata dal numero di parametri liberi $|W|$, dal grado polinomiale $M$, o dalla **norma del vettore dei pesi $\|w\|^2$**.
* La regolarizzazione di Tikhonov $L_2$ (Weight Decay $\frac{\lambda}{2}\|w\|^2$) limita la norma dei pesi, riducendo la complessità *effettiva* e levigando la superficie di risposta senza modificare l'architettura dei neuroni.

#### 3. Complessità Computazionale
* Misura il costo in termini di risorse temporali e spaziali (es. la Backpropagation ha complessità $O(|W|)$ per pattern per epoca).

---

### 2.3 Grafico ASCII del Trade-off di Complessità (Underfitting vs Overfitting vs SRM)

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
         +----------------------------------------------------> Complessità / VC-Dimension
           [UNDERFITTING]      [OTTIMO SRM]       [OVERFITTING]
           (Bias Alto)                            (Varianza Alta)
```

---

## 📌 DOMANDA 3: Equazione/i della Varianza nella Decomposizione Bias-Varianza

### 3.1 Dimostrazione Algebrica Completa della Decomposizione MSE
Sia $y = f(x) + \epsilon$ il target reale con rumore stocastico a media nulla $\mathbb{E}[\epsilon] = 0$ e varianza $\mathbb{E}[\epsilon^2] = \sigma^2$, e sia $h_D(x)$ la predizione del modello addestrato sul dataset $D$.  
Definiamo l'ipotesi media $\bar{h}(x) = \mathbb{E}_D [h_D(x)]$.

Scriviamo l'errore quadratico medio atteso rispetto a tutti i possibili dataset $D$ ed al rumore $\epsilon$:
$$\mathbb{E}_{D,\epsilon}\left[ (y - h_D(x))^2 \right] = \mathbb{E}_{D,\epsilon}\left[ \big( (f(x) - \bar{h}(x)) + (\bar{h}(x) - h_D(x)) + \epsilon \big)^2 \right]$$

Espandendo il quadrato del trinomio $(A + B + C)^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC$:
1. $A^2 = (f(x) - \bar{h}(x))^2 = \text{Bias}(x)^2$
2. $\mathbb{E}_D [B^2] = \mathbb{E}_D [(\bar{h}(x) - h_D(x))^2] = \text{Varianza}(x)$
3. $\mathbb{E}_\epsilon [C^2] = \mathbb{E}[\epsilon^2] = \sigma^2$ (Rumore Irriducibile)
4. I doppi prodotti si azzerano poiché $\mathbb{E}[\epsilon] = 0$ e $\mathbb{E}_D [\bar{h}(x) - h_D(x)] = \bar{h}(x) - \bar{h}(x) = 0$.

Otteniamo la decomposizione fondamentale:
$$\mathbf{\mathbb{E}_{D,\epsilon}[(y - h_D(x))^2] = \text{Bias}(x)^2 + \text{Varianza}(x) + \sigma^2}$$

---

### 3.2 Equazione Esplicita della Varianza
La **Varianza** per un singolo punto $x$ è definita formale come:
$$\text{Varianza}(x) = \mathbb{E}_D \left[ \big( h_D(x) - \bar{h}(x) \big)^2 \right]$$

dove $\bar{h}(x)$ è l'**ipotesi media del modello**, ottenuta mediando le predizioni di modelli addestrati su tutti i possibili dataset $D$:
$$\bar{h}(x) = \mathbb{E}_D [h_D(x)]$$

Integrata sull'intero dominio d'ingresso $\mathcal{X}$ con distribuzione $p(x)$:
$$\text{Varianza Globale} = \int_{\mathcal{X}} \mathbb{E}_D \left[ \big( h_D(x) - \bar{h}(x) \big)^2 \right] p(x) \, dx$$

---

### 3.3 Significato Fisico e Riduzione della Varianza via Bagging
* **Sensibilità al Dataset**: Misura quanto la predizione $h_D(x)$ **oscilla / varia** al variare dello specifico training set $D$ estratto dalla distribuzione $P(x,y)$.
* **Riduzione via Ensemble Bagging**: Facendo la media delle predizioni di $M$ modelli de-correlati con varianza $\sigma^2$ e correlazione d'errore $\rho$:
  $$\text{Varianza}(h_{ens}) = \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2$$
  Se i modelli sono indipendenti ($\rho \to 0$), la varianza dell'ensemble si riduce di un fattore $M$.

---

## 📌 DOMANDA 4: VC-Dimension di un Intervallo con Due Estremi su Dominio Reale ($\mathbb{R}^1$)

### 4.1 Definizione Formale dello Spazio delle Ipotesi $\mathcal{H}$
Consideriamo la classe di ipotesi $h_{[a,b]}(x)$ definite da un intervallo chiuso $[a, b] \subset \mathbb{R}$ su dominio unidimensionale con estremi variabili $a \le b$:
$$h_{[a,b]}(x) = \begin{cases} +1 & \text{se } a \le x \le b \\ -1 & \text{se } x < a \text{ oppure } x > b \end{cases}$$
*(La regione ad etichetta positiva $+1$ è rappresentata da un singolo segmento contiguo $[a,b]$ sulla retta reale).*

---

### 4.2 Passo 1: Shattering di $N = 2$ punti ($VC \ge 2$)
Siano dati due punti distinti ordinati sulla retta reale: $x_1 < x_2$.  
Verifichiamo la capacità di realizzare tutte le $2^2 = 4$ possibili dicotomie binaria:

1. **Dicotomia $(-1, -1)$**: Posizioniamo l'intervallo $[a,b]$ totalmente all'esterno di entrambi i punti (es. $a > x_2$).
2. **Dicotomia $(+1, -1)$**: Posizioniamo l'intervallo $[a,b]$ in modo da contenere solo $x_1$ (es. $a < x_1 < b < x_2$).
3. **Dicotomia $(-1, +1)$**: Posizioniamo l'intervallo $[a,b]$ in modo da contenere solo $x_2$ (es. $x_1 < a < x_2 < b$).
4. **Dicotomia $(+1, +1)$**: Posizioniamo l'intervallo $[a,b]$ in modo da contenere contemporaneamente sia $x_1$ che $x_2$ (es. $a < x_1 < x_2 < b$).

Poiché tutte le 4 combinazioni sono realizzabili, l'insieme di 2 punti è **shatterato**. Pertanto $h_{VC} \ge 2$.

---

### 4.3 Passo 2: Impossibilità di Shatterare $N = 3$ punti ($VC < 3$)
Siano dati 3 punti arbitrari ordinati sulla retta reale: $x_1 < x_2 < x_3$.  
Proviamo a realizzare la dicotomia che assegna etichetta **$+1$ a $x_1$ e $x_3$** e **$-1$ al punto intermedio $x_2$**:
$$\text{Dicotomia richiesta: } \big( x_1 \to +1, \,\, x_2 \to -1, \,\, x_3 \to +1 \big)$$

* Affinché $x_1 \to +1$, l'estremo sinistro deve soddisfare $a \le x_1$.
* Affinché $x_3 \to +1$, l'estremo destro deve soddisfare $b \ge x_3$.
* Poiché i punti sono ordinati $x_1 < x_2 < x_3$, dalle condizioni $a \le x_1$ e $b \ge x_3$ segue che $a \le x_1 < x_2 < x_3 \le b$.
* Di conseguenza, l'intervallo $[a,b]$ conterrà **necessariamente** anche il punto centrale $x_2$, costringendo l'ipotesi ad assegnare $h(x_2) = +1$.
* È dunque **impossibile** assegnare $h(x_2) = -1$ quando sia $x_1$ che $x_3$ hanno etichetta $+1$.

Poiché esiste almeno una dicotomia irrealizzabile per qualsiasi disposizione di 3 punti, l'insieme di 3 punti non può essere shatterato.

---

### 4.4 Conclusione Formale
$$\mathbf{h_{VC} = 2}$$

---

### 💡 Consigli per l'Esposizione all'Orale
Se il Prof. Micheli ti riprende la Domanda 4 all'orale:
> *"Professore, la VC-dimension dell'intervallo chiuso su $\mathbb{R}^1$ è esattamente 2.*  
> *Per $N=2$ punti distinti $x_1 < x_2$, possiamo shatterare l'insieme realizzando tutte le 4 dicotomie regolando opportunamente gli estremi $a$ e $b$.*  
> *Per $N=3$ punti ordinati $x_1 < x_2 < x_3$, se assegniamo etichetta $+1$ ai due punti esterni $x_1$ e $x_3$, l'intervallo chiuso $[a,b]$ deve necessariamente includere anche il punto centrale $x_2$, rendendo impossibile ottenere la dicotomia con $x_2 \to -1$. Pertanto $h_{VC} = 2$."*
