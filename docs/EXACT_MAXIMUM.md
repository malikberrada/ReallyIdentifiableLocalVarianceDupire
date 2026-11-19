# Maximum exact : coefficient à plateau et prix continus certifiés

## Hypothèses et égalité extrême

La normalisation est X0=1 et dX=X sqrt(a(t,X)) dW, avec 0<amin<=a<=amax.
Les forwards et discounts sont fixés, les prix européens sont des intervalles
exacts. La classe ne prescrit pas de régularité temporelle supplémentaire.
Pour phi>=0, D=∫phi k²C_kk et A=∫phi a k²C_kk / D. Le certificat de dénominateur
prouve D>0; les priors impliquent amin<=A<=amax.

Si un modèle compatible vérifie a=amax presque partout sur le support de phi,
**A=amax exactement**. Le résultat n'utilise pas de quadrature de A. Le modèle
fourni a ce plateau sur (5/16,7/16)×(9/10,11/10). Un autre modèle, l'horloge
C² de la branche précédente, atteint amin sur toute la bande temporelle.

## Diffusion définie sur tout le demi-axe positif

Sur chaque panneau spatial [u,v], le coefficient est

    a(k) = a_left + (a_right-a_left) S((k-u)/(v-u)),
    S(z) = 10z³-15z⁴+6z⁵.

S'(z)=30z²(1-z)²>=0, S(0)=0, S(1)=1. Les deux premières dérivées s'annulent aux
extrémités. Les contrôles rationnels sont dans les priors; a est donc C² en
état, positif et borné, y compris par extension constante hors des panneaux.
Il est constant par morceaux en temps, vaut prefix_variance avant prefix_end,
et reprend prefix_variance après la dernière phase. Le coefficient X sqrt(a)
est lipschitzien en état avec croissance linéaire. La diffusion forte existe,
reste positive, a des trajectoires continues et est une vraie martingale sur
chaque horizon fini (variance bornée et Novikov).

Cette classe autorise les sauts temporels du coefficient. Le résultat n'est
pas annoncé pour une classe imposant a C² en temps ou une borne de ses dérivées.

## Certifier les prix par comparaison PDE

La surface vraie C satisfait C_t-a(t,k)k² C_kk/2=0.
La surface candidate V est une interpolation tensorielle Hermite cubique en
t et en k, à partir de valeurs, pentes spatiales, dérivées temporelles et
pentes mixtes binaires stockées. Ces nombres désignent des rationnels exacts.
Les données nodales partagées garantissent V et V_k continus aux jonctions.
Les dérivées temporelles coïncident dans chaque phase; V_t peut sauter aux
changements de coefficient. Les phases sont traitées successivement.

Sur chaque cellule complète, calculer r=V_t-a k² V_kk/2, de degré au plus
3 en t et 8 en k. Ses coefficients de Bernstein donnent un encadrement sur
TOUT le rectangle. Une borne rho_j contrôle le supremum spatial pendant
l'intervalle temporel j. Les jonctions C¹ en espace ne produisent pas de
masse de courbure singulière; la comparaison s'applique à cette surface par
morceaux et continue aux bords des phases.

A t0, C est le Black connu de variance intégrée v=a0*t0. Sur chaque cellule
centrale de largeur h, l'erreur Hermite est au plus

    max(endpoint value errors) + h/4 max(endpoint slope errors)
       + h⁴/384 sup |Black⁽⁴⁾|.

Les valeurs et pentes Black sont encadrées en fractions par le moteur de séries.
Avec b=3/2+log(k)/v,

    Black⁽⁴⁾ = (Black''/k²) (b²+b-1/v).

Dans les ailes, la valeur temps est majorée par un put/call Black au strike de
coupure. On ne fait pas confiance à l'approximation libm de Phi pour ce contrôle.
Cela donne epsilon0.

Les conditions numériques V(t,L)=1-L et V(t,R)=0 ne sont **pas imposées à la
diffusion**. Sa vraie surface est encadrée par positivité, convexité et
comparaison Black de variance totale supérieure

    v_upper(T)=a0*t0+amax*(T-t0).

L'erreur au bord jusqu'à T est au plus la valeur temps du Black à L ou le call
Black à R, avec tout arrondi du bord gauche ajouté : epsilon_boundary(T).
Les valeurs de bord de V sont contrôlées pendant tout l'intervalle.
Le principe du maximum, phase par phase, donne

    |C(T,K)-V(T,K)| <= max(epsilon0,epsilon_boundary(T))
       + sum_j length(j intersect [t0,T])*rho_j.

La somme et l'évaluation de V à la cotation sont rationnelles. L'intervalle de
prix obtenu doit être ENTIÈREMENT contenu dans le spread. La surface V n'est
pas annoncée comme une surface de calls admissible : elle certifie les prix
de la diffusion définie séparément. Les erreurs de solveur, de discrétisation
et de domaine sont absorbées dans le résidu et les bords; aucune différence
entre deux maillages n'est admise comme borne d'erreur.

## Arithmétique et confiance

Le moteur historique interval.py utilise des fractions et des séries avec
reste explicite pour log, exp, sqrt, Phi et Black. Le contrôle massif des
polynômes utilise binary_intervals.py : chaque opération élémentaire est
entourée par numpy.nextafter vers l'extérieur. Cela inclut les poids binomiaux,
les divisions par 3 et les changements de base. Une ufunc séparée est utilisée
pour chaque opération; pas de BLAS, d'expression fusionnée ou de fonction
transcendante approchée dans ce chemin. Les entrées non finies et les divisions
par des intervalles contenant zéro sont rejetées. Les suprema de Bernstein
sont intégrés ensuite en fractions exactes.

Préconditions : IEEE-754 binary64, arrondi au plus proche, sous-normaux graduels.
Le démarrage les contrôle et les tests confrontent les intervalles aux fractions.
La garantie suppose une installation NumPy conforme, sans arithmétique fast-math
non conforme. Ce n'est pas une preuve formalisée dans Lean/Coq. Les coefficients
du candidat sont exacts; leur erreur numérique de calcul n'est jamais ignorée.

La proposition utilise un schéma compact et SDIRK2 L-stable, avec Thomas C++
optionnel ou LAPACK. Aucun ordre de convergence supposé ne remplace la preuve.
Le vérificateur ne relance ni le solveur PDE ni l'optimiseur.

## Frontière sur des ensembles continus

Le même couple de modèles atteint les deux priors pour tout noyau non négatif,
non nul sur un ensemble de mesure positive, contenu dans le rectangle commun.
D>0 résulte de la non-dégénérescence; le noyau initial possède en plus le
certificat numérique uniforme D>=1/8192 importé et revérifié.

Les mélanges convexes des deux lois et de leurs flux gardent tous les prix
dans les spreads. Leur coefficient local est le rapport du flux mélangé à la
densité mélangée; il reste entre les priors. La fonctionnelle du mélange est
un rapport de fonctions affines du poids, continu entre les deux extrêmes.
Toutes les valeurs intermédiaires sont donc également réalisables.
L'intervalle d'identification de cette région est tout le prior. L'erreur
minimax absolue est (amax-amin)/2=99/800 : les deux témoins imposent la borne
inférieure et le milieu du prior atteint la borne supérieure.

Le domaine de fenêtres est un produit de quatre intervalles rationnels fermés
(t,k,hT,hK), entièrement dans le domaine physique. Pour chaque boîte, le code
contrôle symboliquement que TOUS ses supports sont dans une région prouvée.
L'atlas peut également utiliser les bandes et horloges exactes de 0.3.0.
Une boîte non couverte est subdivisée; l'arbre conserve parent, axe, point de
coupe et enfants. Le vérificateur reconstruit la partition : pas de trou ou de
fausse inclusion accepté. Une feuille sans théorème reste ouverte à budget épuisé.

Le domaine par défaut contient tous les hT entre 1/64 et 1/16 et tous les hK
entre 1/20 et 1/10, à t=3/8,k=1. Un autre exemple fait varier les quatre
coordonnées. Il s'agit de preuves sur des continua, sans extrapolation d'une
grille finie. **Cela ne résout pas tous les paramètres positifs sans domaine
borné**, tous les smiles ou tous les spreads. Aucune fermeture ni vitesse de
convergence universelle n'est annoncée pour une entrée arbitraire.

## Antériorité

Les formulations convexes par transport existent déjà : Guo, Loeper et Wang,
Local Volatility Calibration by Optimal Transport,
https://arxiv.org/abs/1709.08075. Les mesures d'occupation locales sont étudiées
par Holtorf, Edelman et Rackauckas, https://arxiv.org/abs/2211.15652.
Ces articles ne sont pas présentés comme des preuves du cas fourni ici.
Comparaison PDE et Bernstein sont des outils classiques. Une preuve générale
de nouveauté scientifique et une validation sur marché restent distinctes.
