# Définition de la classe et portée des preuves

On fixe des courbes déterministes F,D, puis X=S/F, X0=1. La classe est celle des
solutions martingales de dX=X sqrt(a(T,X)) dW avec 0<amin≤a≤amax<∞. Les calls
sont c(T,k)=E[(X_T−k)+]. Les coefficients peuvent être mesurables, sous les
conditions usuelles d'existence ; une formulation faible est utilisée. Une
variance bornée assure les moments et la propriété de vraie martingale. Les
intervalles de cotation sont des entrées exactes, non des intervalles de
confiance statistiques. La non-vacuité est une question distincte.

## Moyenne étudiée

Pour phi≥0, C², à support compact intérieur :

    A_phi = ∫phi a k² c_kk / ∫phi k² c_kk = N_phi / D_phi,
    N_phi = -2∫phi_T c,    D_phi = ∫(k²phi)_kk c.

Les intégrales sont en T,k. Le poids dépend de la densité propre au modèle.
Cela n'est pas une moyenne uniforme de a, ni une estimation de a en un point.
Une interprétation ponctuelle exige des hypothèses supplémentaires de régularité.
Le noyau de la bibliothèque est le produit de (1−u²)^3 sur |u|<1, nul ailleurs.
Il est C² aux extrémités. Les dérivées et intégrales sont polynomiales exactes.

## Encadrement continu P1

Monotonicité et bornes de calls :

    (1−k)+ ≤ c(T,k) ≤ 1,
    -1 ≤ c_k ≤ 0,    c_T ≥ 0.

Pour u≤v, la formule d'Itô et Grönwall donnent

    E[X_u²] ≤ exp(amax*u),
    E[(X_v−X_u)²] ≤ exp(amax*v)−exp(amax*u).

Ainsi c est 1-Lipschitz en strike et son incrément temporel est borné par la
racine du dernier membre. Les deux ancres d'une cotation (Ti,ki,bi,ui) donnent
sur un rectangle [ta,tb]×[ka,kb] :

    c ≥ bi − (kb−ki)+ − sqrt((exp(amax*Ti)−exp(amax*ta))+),
    c ≤ ui + (ki−ka)+ + sqrt((exp(amax*tb)−exp(amax*Ti))+).

Le terme temporel n'est utilisé que lorsque l'ordre des dates le requiert.
L'intersection des enveloppes et des bornes intrinsèques couvre chaque
surface admissible, même entre les contrats. Les corrélations entre cellules
sont abandonnées : cela élargit l'ensemble, sans exclure un modèle admissible.

Sur une cellule, c∈[l,u]. Pour un poids polynomial w de signe constant,
∫wc est borné en multipliant l/u par son intégrale exacte. Lorsque son signe
n'est pas certifié, on utilise

    ∫wc = l∫w + ∫w(c−l),
    (u−l) min(0,inf w) |cell| ≤ ∫w(c−l)
                            ≤ (u−l) max(0,sup w) |cell|.

Le minimum et maximum du poids sont encadrés par les coefficients de
Bernstein, pas par échantillonnage. Les cellules de calcul supplémentaires
réduisent cet élargissement. Leur taille n'est pas la résolution identifiable.
Les encadrements obtenus à plusieurs niveaux sont intersectés. Une borne
restant large n'est pas, à elle seule, une preuve d'impossibilité.

Si D∈[dl,du] et N∈[nl,nu], D>0, alors

    A ≥ max(amin,nl/du),
    A ≤ min(amax,nu/dl) si dl>0 ; sinon la borne haute reste amax.

`denominator_unresolved` signifie que le calcul n'a pas fourni de minorant
quantitatif strict. La positivité qualitative dans cette classe elliptique ne
rend pas une division par un minorant numérique nul légitime. Les noyaux
compacts n'exigent aucune troncature d'une diffusion aux bords de la grille.

Cet encadrement est volontairement conservateur. Il n'utilise pas toutes les
dépendances convexes de la classe et ne constitue pas une solution sharp de P1.

## LP fini et preuve duale

Les appels sont des variables c_ij, avec contraintes de convexité sur strikes
irréguliers, pente dans [-1,0], monotonie temporelle, bornes de prix, données
bid–ask et inégalités Dupire en Euler implicite. Les contrats sont des nœuds
exacts de cette grille numérique. La frontière droite est libre entre ses
bornes de prix. Le ratio discret est une quadrature des incréments temporels
et gammas au temps droit ; ce n'est pas exactement le ratio continu.

Après min D discret certifié strictement positif, Charnes–Cooper utilise
y=x*s, d.y=1, s=1/(d.x). La borne s≤1/dmin est arrondie vers le haut.

Pour tout LP min f.x, Ax≤b, l≤x≤u et tout lambda≥0, même inexact :

    min f.x ≥ −lambda.b + Σ_j min(r_j*l_j, r_j*u_j),
    r = f + A^T lambda.

Cette formule est vérifiée avec `Fraction.from_float`, donc sur les valeurs
binaires exactes sauvegardées. Aucun résidu de stationnarité n'est omis. La
solution primale est également vérifiée : une violation de 10^-15 reste une
violation exacte, jamais transformée en « témoin certifié » par une tolérance.
Une éventuelle infeasibilité HiGHS est marquée numérique, sans théorème de
Farkas ajouté implicitement.

## Témoins analytiques et limite minimax

Entre deux maturités observées consécutives L<R :

    g(t)=sin(2*pi*(t−L)/(R−L)) dans (L,R), 0 ailleurs,
    a_±(t,k)=a0 ± e*g(t).

La variance est continue, positive et bornée sous les conditions affichées.
La solution est lognormale à variance intégrée V_±(T). À toutes les maturités
observées, V_±(T)=a0*T exactement, puisque ∫_L^R g=0 et aucune observation ne
tombe strictement entre L et R. Les prix des deux modèles sont alors identiques
pour **tous** les strikes à ces maturités.

Les prix Black c=Phi(d1)−k Phi(d2) sont vérifiés dans les spreads au moyen
d'intervalles rationnels. Pour une fenêtre contenue dans la demi-vague positive,
g≥m>0 sur son support. La positivité des poids implique

    A_phi(a_−) ≤ a0−e*m,    A_phi(a_+) ≥ a0+e*m.

L'erreur maximale de tout estimateur basé sur les mêmes observations est donc
au moins e*m. La pondération différente des deux modèles ne compromet pas cet
argument, qui borne directement chaque moyenne. Cette séparation n'atteste
pas que les témoins atteignent les extrémités de la classe entière. La famille
déterministe ne calibrera pas en général une chaîne réelle avec smile.

## Arithmétique des transcendantes

Tout est effectué avec fractions exactes, puis compression dyadique vers
l'extérieur à 110 bits. exp(x) utilise sa série positive et une majoration
géométrique du reste, avec inversion pour x<0. log(x) utilise une réduction
par puissances de 2 puis la série de atanh. sqrt utilise une racine entière
de rationnels mis à l'échelle. Pi utilise la formule de Machin avec reste
alterné d'arctangente. Phi intègre la série de exp(−u²/2) avec reste de Lagrange ;
pour x≥12, l'inégalité de Mills donne l'encadrement de la queue.

Ces preuves ne dépendent ni de l'arrondi de libm, ni d'un calcul GPU, ni d'une
simple comparaison entre deux précisions flottantes. Elles dépendent de la
correction du code rationnel et des arguments exposés ici ; il n'y a pas de
vérification formelle externe.

## P2 par principe du maximum

Sur [0,Tmax]×[ell,R], pour une candidate polynomiale v, on certifie

    rho ≥ sup |v_T − a k² v_kk/2|,
    delta0 ≥ sup |v(0,k)−(1−k)+|,
    deltaB ≥ sup_bords |v−c|.

À gauche, c(t,ell)∈[1−ell,1]. À droite, pour X≥0,
(X−R)+≤X²/(4R), d'où c(t,R)≤min(1,exp(amax*Tmax)/(4R)). Cela contrôle
explicitement le bord et la queue, sans imposer une valeur arbitraire à c.
Le principe du maximum, avec les barrières ±[max(delta0,deltaB)+rho*t], donne

    |c(Ti,ki)−v(Ti,ki)| ≤ eta,
    eta=max(delta0,deltaB)+Tmax*rho.

Le résidu est encadré sur toutes les cellules par Bernstein rationnel. Pour
a variable quelconque dans [amin,amax], il suffit de borner les deux résidus
polynomiaux aux extrémités amin/amax : le résidu dépend affinement de a.
L'estimation est donc uniforme pour cette classe, mais peut être très large.
Pour un a polynomial fixé, ses bornes sont vérifiées à l'intérieur du domaine ;
le résultat couvre toute extension extérieure conservant amin≤a≤amax.

Si bid_i+eta≤v(Ti,ki)≤ask_i−eta, les prix continus sont compatibles. Rien dans
cette version ne garantit qu'une reconstruction polynomiale avec petits
résidus existe pour toute solution LP. Le théorème de raffinement général
demandé par P2 reste à établir, même si les encadrements utilisés ici sont valides.

## Références vérifiées pour situer le travail

1. Rückert, Anderssen, Hofmann (2012), *Stable Parameter Identification
   Evaluation of Volatility*. Formulation faible et localisation déjà connues :
   https://www.tu-chemnitz.de/mathematik/preprint/2012/PREPRINT_03.pdf
2. Achdou, *An Inverse Problem for a Parabolic Variational Inequality Arising
   in Volatility Calibration with American Options*, DOI
   https://doi.org/10.1137/S0363012903424423
3. HiGHS via SciPy, multiplicateurs, tolérances et moteur C++ :
   https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
4. CuPy, roues CUDA12/13, NVRTC : https://docs.cupy.dev/en/latest/install.html
5. Setuptools, extension C/C++ directe :
   https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
6. Cboe, SPX européen et conventions AM/PM :
   https://www.cboe.com/tradable-products/sp-500/spx-options
7. Cboe, XSP européen :
   https://www.cboe.com/tradable_products/sp_500/mini_spx_options/specifications

Le lien CERMICS fourni dans la demande n'a pas pu être récupéré de manière
fiable pendant cette implémentation. Aucun théorème spécifique de ce rapport
n'est annoncé comme implémenté ou redémontré. Cette documentation n'est pas
une revue bibliographique systématique ni une certification de nouveauté.
