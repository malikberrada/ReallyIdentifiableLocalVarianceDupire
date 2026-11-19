# Certification continue dans la version 0.2.0

## Problème et portée

Les observations sont des calls européens normalisés, X0=1, avec courbes de
forward et d'actualisation fixées. Le processus est une martingale positive
continue dX = sqrt(a(t,X)) X dW, avec 0 < amin <= a <= amax. Les coefficients
peuvent être irréguliers côté extérieur ; les témoins construits sont réguliers.
La garantie est conditionnelle à ces hypothèses et aux intervalles de prix
fournis. Elle n'englobe pas l'incertitude des courbes ni les prix américains.

On fixe φ(t,k)=pT(t)pK(k), avec p(x)=(1-((x-centre)/demi-largeur)^2)^3
sur son support et zéro ailleurs. La quantité est A=N/D, où

    N = ∫ φ(t,k) a(t,k) k² µt(dk) dt,
    D = ∫ φ(t,k) k² µt(dk) dt.

Lorsque la loi a une densité, µt(dk)=c_kk(t,k) dk. Le poids dépend du modèle.
Toutes les largeurs et tolérances sont en variance annualisée.

## Relaxation extérieure qui couvre les diffusions

Le module `occupation.py` utilise des variables de probabilité, premier et
deuxième moments dans chaque cellule spatiale, aux dates de partition. Il
retient aussi leurs mesures d'occupation, pondérées par les polynômes de
Bernstein positifs de degré 1 ou 2 en temps. Une variable de flux représente
l'intégrale de a*x² contre cette mesure ; elle est encadrée par amin et amax
fois le deuxième moment. La dernière cellule est [R,+infini). Aucune condition
absorbante ou réfléchissante n'y est introduite.

Pour r>=2, Itô et Grönwall donnent E[Xt^r] <= exp(r(r-1)amax*t/2).
Cela donne des bornes finies pour chaque variable, y compris dans les queues.
E[Xt²] >= exp(amin*t). Les bornes de moments dans une cellule, les tangentes
linéaires de S²<=P*H, la masse totale 1 et l'espérance 1 sont nécessaires pour
toute mesure physique. Ces tangentes ne sont pas présentées comme une
description exacte du cône des moments.

Les strikes des observations sont des frontières de cellules. Le prix call
est donc exactement la somme S-KP des cellules situées au-dessus de K, queue
infinie comprise. On ajoute bid/ask, monotonie calendaire et comparaison globale
avec Black aux variances amin et amax.

Pour chaque test spatial f lisse, Itô donne les identités faibles correspondant
à f, (t-Ta)f et (Tb-t)f sur chaque tranche temporelle. Les tests sont des calls
et digitals lissés dont f'' a un support compact. Une interpolation quadratique
de f est intégrée avec les trois moments ; son reste est borné exactement par
Bernstein sur la cellule. f'' est aussi encadré sur toute la cellule. Les deux
côtés de l'identité deviennent des inégalités nécessaires, jamais une équation
de différences finies supposée exacte.

Pour la cible, les variables Z et Y représentent les intégrales exactes de
B(t)*φ*x² et B(t)*φ*a*x². Les minima et maxima polynomiaux certifiés de φ sur
chaque cellule les relient aux moments et au flux. La partition de l'unité des
polynômes B assure D=sum Z et N=sum Y. La relaxation peut être large mais tout
modèle admissible s'y projette.

## Contraintes directes issues des écarts à Black

Pour un call observé en (T,K), le prix Black rétrospectif de variance constante
amin satisfait, par Itô le long du modèle :

    c(T,K)-c_BS(amin;T,K)
      = 1/2 E ∫0^T (a-amin) Xt² Γ_BS(amin;T-t,Xt,K) dt.

L'analogue avec amax donne c_BS(amax)-c = 1/2 E∫(amax-a)X²Γmax.
Les intégrandes sont positifs. Pour T strictement après le support temporel
de φ, des encadrements rationnels sur des boîtes couvrant le support donnent
une constante B vérifiant φ <= B*Γ. On en déduit les coupes nécessaires :

    N-amin*D <= 2B*(ask-c_BS(amin).lo),
    amax*D-N <= 2B*(c_BS(amax).hi-bid).

Le gamma rétrospectif est K*G(T-t,x/K)/x², où G(y)=y²c_kk(y) est la densité
pondérée Black. Si le minorant de gamma utilisé est nul, la coupe est omise.
Une petite constante non justifiée ne lui est jamais substituée. Les constantes
et budgets exacts sont enregistrés dans `flux_budget_cuts`.

## Passage au LP et dual exact

Toutes les variables physiques sont positives. Chaque ligne rationnelle est
divisée par une constante positive, puis chaque coefficient est arrondi vers
-infini, chaque second membre et chaque borne supérieure vers +infini. Ainsi
la version binaire contient la projection physique. Le vérificateur reconstruit
ces lignes depuis les observations et contrôle les arrondis un par un.

HiGHS fournit uniquement des multiplicateurs proposés. Pour Ax<=b, l<=x<=u,
et lambda>=0, le certificat de min f*x est

    -lambda*b + sum_j min(rj*lj,rj*uj),   r=f+A'lambda.

Ce calcul est effectué en fractions exactes, avec l'objectif rationnel
d'origine. Aucun résidu de stationnarité n'est supprimé. Même une proposition
non optimale donne une borne valide, éventuellement faible.

On minore D par ce procédé. Si le minorant est positif, Charnes–Cooper introduit
y=x/D et s=1/D, avec s<=1/Dmin. Les lignes projetées, les bornes y<=u*s et
d*y=1 sont construites rationnellement, puis à nouveau arrondies vers
l'extérieur. Les duals du minimum et du maximum donnent l'intervalle continu.
Si Dmin n'est pas positif, l'état `denominator_unresolved` conserve les priors.

Le primal de ce LP est une relaxation de mesures. Il n'est jamais converti
directement en diffusion ni utilisé comme témoin admissible.

## Témoins continus par splines compactes

Le point de départ est un call analytique c0=sum wj*Black(aj), avec wj>=0,
sum wj=1 et amin<aj<amax, dont tous les prix sont contenus dans les spreads.
Les poids peuvent être recherchés numériquement, mais leur compatibilité est
ensuite contrôlée en arithmétique exacte et par intervalles.

On cherche c=c0+P, où P est une combinaison rationnelle de B-splines cubiques
tensorielles à support compact en temps et strike, loin de t=0 et k=0.
Les splines et leurs deux premières dérivées se raccordent à zéro. On exige
sur tout le support :

    G0+k²Pkk >= beta*G0 > 0,
    2*c0T+2*Pt-amin*(G0+k²Pkk) >= 0,
    amax*(G0+k²Pkk)-2*c0T-2*Pt >= 0.

Les termes polynomiaux ont une représentation de Bernstein exacte. On ajoute
la borne inférieure analytique de G0, ou de ses combinaisons positives, sur
chaque sous-boîte. La positivité de tous les coefficients est une condition
suffisante partout, et non seulement aux nœuds.

Alors a=2*cT/(k²*cKK) est comprise entre amin et amax. La densité est strictement
positive, les conditions initiales et les queues sont celles du mélange, et la
masse et l'espérance restent 1. Le coefficient est localement lipschitzien sur
le support compact et analytique à l'extérieur. La diffusion correspondante
est positive, non explosive et martingale sous la borne supérieure de variance.
Sa loi résout l'équation de Fokker–Planck de c_kk ; les conditions initiales et
l'unicité donnent précisément les prix c. Il n'y a donc pas d'erreur de PDE à
ajouter pour ces témoins.

Les inégalités de Bernstein et les prix ont un point faisable connu : P=0.
Après l'optimisation flottante, chaque coefficient est converti en fraction
binaire exacte. Une contraction rationnelle globale vers zéro répare toutes
les lignes, avec une marge stricte. Toutes les contraintes sont ensuite
recalculées exactement. Cette réparation reste valable si la recherche s'est
arrêtée sur sa limite de temps. Le certificat distingue la faisabilité exacte
de la convergence numérique de l'optimiseur.

Les petits problèmes sont résolus en LP fractionnaire complet. Pour les grands,
une recherche de Dinkelbach avec ajout des contraintes violées conserve la base
simplex dans le moteur C++ public de HiGHS. Les lignes non sélectionnées sont
toutes contrôlées par la réparation finale ; cette accélération ne réduit pas
le domaine de vérification.

## Quadrature validée de la moyenne du témoin

Les intégrales de φ*Pt et φ*k²Pkk sont polynomiales et exactes. Pour la partie
Black, `density.py` encadre G, son gradient et sa Hessienne par intervalles.
Une quadrature de Taylor au milieu utilise les moments exacts du noyau et un
reste quadratique borné sur chaque cellule. Elle est intersectée avec le
simple encadrement de G sur la cellule. Cela fournit une preuve d'erreur,
indépendante d'une différence entre deux grilles. Le dénominateur du modèle
est aussi minoré par beta fois le minorant du dénominateur de base.

## Profils temporels atteignant une extrémité

Lorsque les prix sont compatibles avec un même Black et que le noyau se trouve
dans un trou entre deux maturités, une variance déterministe C² peut être
constante sur le support du noyau. Des rampes quintiques positives et une
compensation extérieure préservent exactement l'intégrale de variance à toutes
les dates observées. Les prix observés ne changent pas. La moyenne cible est
exactement la valeur du plateau. Lorsque cette valeur vaut amin ou amax, le
modèle atteint l'extrémité globale correspondante. Les coefficients de
Bernstein vérifient les priors et les raccords C² sont vérifiés exactement.

## Ce que signifie la proximité annoncée

Si L<=inf A et sup A<=U sont des bornes extérieures, et si les modèles construits
donnent Aminus dans [lm,um] et Aplus dans [lp,up], alors

    distance du modèle bas à inf A <= um-L,
    distance du modèle haut à sup A <= U-lp.

Ces deux nombres sont calculés avec les intervalles certifiés. Le statut
`endpoints_within_tolerance` n'est produit que s'ils sont tous deux sous la
tolérance. La séparation max(0,lp-um) est une borne inférieure de la largeur
d'identification ; sa moitié est une borne minimax pour une observation limitée
aux mêmes intervalles bid/ask, selon la perte absolue.

L'intersection des bornes extérieures et le choix des meilleurs témoins ne
peuvent dégrader ces garanties entre les niveaux demandés. Ce fait ne prouve
pas la convergence des deux familles vers toutes les extrémités de la classe,
ni un taux, ni un arrêt à toute précision. Les mélanges Black compatibles sont
une limitation de l'initialisation actuelle. Aucun théorème universel de
frontière atteignable n'est annoncé dans cette version.

## Évaluer les modèles et vérifier le raffinement fourni

Après `verify-tighten`, un modèle exporté s'évalue sans PDE numérique :

```python
import json
from pathlib import Path
from dupire_identify.models import ContinuousModel

doc = json.loads(Path("runs/strong/minimum_model.json").read_text(encoding="utf-8"))
model = ContinuousModel(doc)
print(model.call("3/4", "1").json())
print(model.local_variance("3/8", "1").json())
```

Les deux témoins supplémentaires de `examples/refined_witnesses/models.json`
ont été construits avec 24 cellules par axe, subdivisions 2, quadrature 32 et
budget numérique 60 secondes par extrémité. Leur modèle haut a une moyenne
dans [0,22168300186 ; 0,22174003822], donc une distance au supremum global au
plus 0,02831699814 sous le prior amax=0,25. La recherche numérique n'avait pas
terminé à la limite de temps ; la certification primale exacte a néanmoins
abouti. Pour les revérifier individuellement :

```python
from dupire_identify.inner import verify_witness
from dupire_identify.io import payload_quotes
from dupire_identify.continuous import Window

docs = json.loads(Path("examples/refined_witnesses/models.json").read_text(encoding="utf-8"))
for doc in docs:
    print(verify_witness(doc, payload_quotes(doc["quotes"]), Window.make(**doc["window"])))
```

Cette vérification individuelle prend pour hypothèses les observations et
priors de chaque document. La commande `verify-tighten` assure en plus leur
liaison avec les observations et la classe déclarées dans un calcul complet.

## Sources et vérification

Les formulations convexes densité/flux, le transport optimal et les conditions
de non-arbitrage de Bernstein ont des antériorités. Cette implémentation ne
les présente pas comme nouveaux :

- [Guo, Loeper et Wang, Local Volatility Calibration by Optimal Transport](https://arxiv.org/abs/1709.08075).
- [HiGHS : interface Python publique](https://ergo-code.github.io/HiGHS/stable/interfaces/python/).
- [HiGHS : options du solveur](https://ergo-code.github.io/HiGHS/dev/options/definitions/).

`verify-tighten` reconstruit la projection, les LP et les témoins depuis les
observations, vérifie les sommes de contrôle et recalcule les preuves exactes.
Il ne relance pas le solveur. Le code rationnel et les séries transcendantes
constituent la base de confiance ; ce n'est pas une preuve formalisée dans un
assistant de théorèmes.
