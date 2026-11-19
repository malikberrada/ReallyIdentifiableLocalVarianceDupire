# Frontière générale : définitions et preuves

## Classe et grandeur

Après normalisation, `dX_t=X_t*sqrt(a(t,X_t))*dW_t`, `X_0=1`,
`0<amin<=a<=amax`. Les observations sont des calls européens dans leurs spreads.
La fonctionnelle est

\[
A_a(\phi)=N_a(\phi)/D_a(\phi),\quad
N_a=\int\phi a k^2p_t(k)\,dt\,dk,\quad D_a=\int\phi k^2p_t(k)\,dt\,dk.
\]

Les diffusions intérieures ont des profils quintiques en état, constants hors
d'un compact, et sont constantes par morceaux en temps. Le coefficient de
diffusion est globalement lipschitzien en état et à croissance linéaire; les
solutions sont positives et de vraies martingales. La non-dégénérescence en log
assure `D>0` pour les bosses positives non nulles à support compact positif.
Un **minorant numérique** strict est une affirmation supplémentaire, calculée
séparément. Les bornes extérieures valent sur une classe plus large : la réciproque
« solution du LP = diffusion » n'est ni supposée ni prouvée.

## Familles et minorants

`families.py` traite une boîte fermée de paramètres réels, pas un échantillonnage.
Les poids sur chaque cellule physique sont encadrés pour toutes ses fenêtres.
Le LP d'occupation conserve masse, moyenne et seconde puissance jusque dans la
queue infinie. Les prix et équations de Dupire faible sont couplés. Les coupes
de flux Black emploient le majorant familial des poids.

Quand les noyaux ont un cœur commun, `phi_w>=c*phi_ref`. En une dimension,
`alpha=delta/hmin`, `beta=href/hmin`, `0<=alpha`, `0<beta<1`,
`alpha+beta<=1` et `|z|<=1` donnent

\[
1-(\alpha+\beta |z|)^2\ge
\left(1-\frac{\alpha^2}{1-\beta}\right)(1-z^2).
\]

En effet, Jensen donne `(alpha+beta*abs(z))^2<=alpha^2/(1-beta)+beta*z^2`;
`c=1-alpha^2/(1-beta)>=beta` termine l'inégalité. Le rayon choisi dans le code
assure `alpha+beta<=1`. Élévation au cube et produit des deux dimensions donnent
une domination rationnelle. Le plancher continu
historique de `D_ref` fournit celui de toute la famille. Sans cœur commun, un
dual LP peut fournir un plancher; sinon le prior est conservé et la famille
peut être subdivisée.

## Objectifs signés et classe vide

Un certificat exact `N-rD>=v`, avec `D>=d>0`, implique

\[
A\ge r+\min(0,v)/d.
\]

Le certificat `rD-N>=v` donne `A<=r-min(0,v)/d`. Les multiplicateurs numériques
sont seulement des propositions; tous les résidus de stationnarité et leurs
corrections par les bornes de variables sont recalculés rationnellement.
La bissection propose des objectifs, jamais des bornes intérieures physiques.

Une Phase I minimise l'écart uniforme aux inégalités. Une borne duale strictement
positive exclut le polytope extérieur et toute diffusion. Une valeur nulle ne
prouve pas l'existence d'une diffusion. Les contradictions simples de monotonie,
convexité, calendrier et prior sont également vérifiées exactement.
Les raffinements conservent les meilleures bornes déjà obtenues; leur complétude
asymptotique n'est pas démontrée.

## Prix continus des modèles intérieurs

La calibration FD propose directement les contrôles de variance. C++ OpenMP
calcule les lots de trajectoires nécessaires aux Jacobiennes, sinon LAPACK est
utilisé. Le coefficient définit le modèle indépendamment du maillage numérique.
La preuve porte sur une reconstruction bicubique : résidus Bernstein avec
arrondis extérieurs, comparaison parabolique, erreurs initiales et de bord.
Les enveloppes Black utilisent les minima et maxima du **coefficient**, jamais
les spreads. Tous les intervalles doivent être inclus dans les cotations.
La compression temporelle est suivie d'une nouvelle preuve sur les grandes
cellules; l'erreur de la grille fine n'est pas réutilisée sans contrôle.

L'intégration par parties donne

\[
N=-2\int\phi_t C,\qquad D=\int(k^2\phi)_{kk}C.
\]

Moments polynomiaux rationnels et sommes dirigées intègrent la reconstruction.
Si `|C-V|<=epsilon`, pour nos bosses,

\[
|N(C)-N(V)|\le(128/35)h_K\epsilon,\qquad
|D(C)-D(V)|\le(32/35)h_T\epsilon\int|\partial_{kk}(k^2b_K)|\,dk.
\]

L'intégrale absolue est majorée par subdivisions de signe Bernstein. Le quotient
est utilisé seulement si son dénominateur est strictement positif; sinon la
plage certaine du coefficient reste disponible.

Pour une famille, si `delta>=sup|phi_w-phi_0|`,
`e=delta*(t_right-t_left)*exp(amax*t_right)` majore la variation de masse.
Avec `d0-e>0`,

\[
|A(\phi_w)-A(\phi_0)|\le (a_{max}-a_{min})e/(d_0-e).
\]

Cette borne est intersectée avec la plage du coefficient sur l'union des supports.
Une plage constante donne une identité exacte. À dénominateur ou budget
arithmétique insuffisant, seule cette plage est conservée.

## Régions générales et coefficients extrêmes explicites

Un modèle compatible **vérifié** suffit pour la région future : si le support
temporel commence après `T_last`, remplacer sa variance par `amin` ou `amax`
sur ce support, pour tous les états, ne modifie aucun prix observé. La moyenne
vaut exactement le prior choisi, sans restriction sur le support spatial positif.

Avec une marge certifiée `m>0` à l'intérieur de tous les spreads, la même
construction fonctionne sur un intervalle de temps assez court, où qu'il soit.
Les processus sont identiques avant cet intervalle. Pendant une modification de
largeur `w`, l'isométrie d'Itô, `E[X_t^2]<=exp(amax*t)` et l'inégalité triangulaire
donnent `E|X_s-Y_s|<=2*sqrt(amax*exp(amax*T_last)*w)` aux dates observées.
Après l'intervalle, le semi-groupe martingale unidimensionnel de base conserve la
constante de Lipschitz 1. Les payoffs call sont eux-mêmes 1-lipschitziens. Ainsi

\[
|C_i^{new}-C_i^{base}|\le2\sqrt{a_{max}e^{a_{max}T_{last}}w}\le m/2
\quad\text{si}\quad
w\le\frac{m^2}{16a_{max}e^{a_{max}T_{last}}}.
\]

Cela conserve simultanément tous les spreads. Il n'y a pas d'amplification
artificielle `exp(L_sigma^2*T)` après modification. Le code utilise un majorant
rationnel d'exponentielle, ou une puissance de deux symbolique pour les grandes
énergies. `construct-general` exporte les deux coefficients et leurs intervalles
de prix, sans résoudre une PDE sur un pas artificiellement minuscule. La
positivité de `D` y est qualitative : `numeric_denominator_lower=null` si aucun
plancher numérique n'a été calculé. Les deux modèles peuvent dépendre de la
fenêtre choisie. Cette taille est suffisante, souvent très conservatrice; elle
ne constitue pas la transition optimale de la frontière.

La conservation Lipschitz est un outil existant : Lowther,
*Properties of Expectations of Functions of Martingale Diffusions* (2008),
[arXiv:0801.0330](https://arxiv.org/abs/0801.0330). Voir aussi Beiglböck, Pammer et
Schachermayer, *From Bachelier to Dupire via optimal transport*, Finance and
Stochastics 26, 59–84 (2022), [DOI](https://doi.org/10.1007/s00780-021-00466-3).
Le projet ne revendique pas ces résultats comme nouveaux.

## Fermeture, partitions et vérification

Avec les bornes extérieures `L,U` et des modèles donnant `[li,ui]` sur toute la
famille, les écarts sûrs sont `min_i(ui)-L` et `U-max_i(li)`. Une boîte n'est
fermée que si les deux sont au plus la tolérance et si un modèle vérifié existe.
Les régions analytiques donnent des extrémités exactes aux priors. Les échecs
numériques gardent les priors et un motif explicite.

Le vérificateur reconstruit les boîtes, coupes, deux enfants de chaque partition,
liens aux observations, duals, résidus et conclusions. Il ne relance pas les
optimiseurs. Une modification d'une affirmation, même après recalcul de son
empreinte, ne suffit donc pas à fabriquer une preuve.

## Épuiser les paramètres positifs

En coordonnées positives `u=T-hT`, `v=K-hK`, `hT`, `hK`, les boîtes
`B_j=[2^-j,2^j]^4` couvrent l'espace quand `j` parcourt **tous** les entiers `>=0`.
Tout quadruplet appartient à une boîte : prendre `j` assez grand pour majorer
chaque coordonnée et son inverse. Un préfixe fini ne couvre pas cette union.

Une résolution universelle demanderait encore de prouver la complétude d'une
relaxation, la densité constructive des modèles compatibles, un contrôle des
queues uniforme sur les classes pertinentes et une terminaison aux frontières
dégénérées. Cette version ne fournit pas ces théorèmes.
