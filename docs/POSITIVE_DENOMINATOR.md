# Minorant strictement positif et certification des extrémités — 0.3.0

## Grandeur et classe

X0=1, dX=X sqrt(a) dW, 0<amin<=a<=amax, prix européens normalisés avec courbes
fixées. Le noyau φ est le produit des deux bosses polynomiales C² de degré 6
du paquet, à support compact dans t>0 et k>0. On considère

    D = E ∫ φ(t,Xt) Xt² dt,
    N = E ∫ φ(t,Xt) a(t,Xt) Xt² dt,
    A = N/D.

Le poids est la loi du modèle, et non la mesure de Lebesgue uniforme en strike.
Les certificats ci-dessous couvrent même les variances adaptées non markoviennes
dans ces bornes, donc en particulier toute la classe de diffusions annoncée.
Ils n'utilisent pas une densité calibrée particulière comme minorant universel.

## Un plancher positif indépendant du LP

En posant M=∫sqrt(a)dW et Q=∫a dt, Dambis–Dubins–Schwarz donne
X_t=exp(B_Qt-Qt/2), avec amin*t<=Qt<=amax*t. On choisit un rectangle intérieur
[t1,t2]×[L,U] du support du noyau, sur lequel φ>=m>0.

Prenons un intervalle rationnel [A,B] contenu dans [log L,log U], et son demi
intervalle central J=[A+(B-A)/4,B-(B-A)/4]. Soient s=amin*t1, v=amax*t2,
Z_u=B_u-u/2 et w=B-A. L'événement

    Z_s dans J, et Z_u dans [A,B] pour tout u dans [s,v]

implique X_t dans [L,U] pour tout t dans [t1,t2], pour **toute** horloge
stochastique admissible. Il n'y a aucune hypothèse d'indépendance entre B et Q.

Le générateur de Z est (∂zz-∂z)/2. La fonction
h(z)=exp(z/2)*sin(π(z-A)/w), nulle aux bords, a pour valeur propre
lambda=π²/(2w²)+1/8. Itô arrêté à la sortie de [A,B] donne

    P_z(survie jusqu'à u) >= exp(-lambda*u)*h(z)/exp(B/2).

Sur J, sin>=1/2. Une borne inférieure explicite de la densité gaussienne de
Z_s sur J donne alors

    D >= (t2-t1)*L²*m*w/[4*sqrt(2πs)] * exp(-E),

    E = max_{z dans J}(z+s/2)²/(2s)
        + lambda*(v-s) + (B-inf J)/2.

Toutes les constantes sont encadrées rationnellement. Pour préserver une
positivité réelle même sous les très petites échelles, le code utilise
exp(2)<8, vérifié par série, donc exp(-E)>=2^(-ceil(3E/2)). Il n'arrondit jamais
une borne négative à une petite constante positive. Le choix entre douze
rectangles est fait par comparaison exacte de leurs bornes.

Sur la fenêtre initiale T=3/8, K=1, hT=1/16, hK=1/10 et les priors
[1/400,1/4], ce seul argument fournit D>=2^-39. Le certificat est recalculé
sans optimisation. Des limites explicites de taille des entiers et de
résolution des logarithmes s'appliquent aux paramètres extrêmes ; dépasser
ces limites déclenche une erreur, pas une preuve de positivité fictive.

## Plancher plus utile grâce aux cotations

Choisissons une maturité observée s antérieure au support temporel du noyau,
et H=T+hT/2, donc avant la fin de ce support. La contribution après H est
positive et peut être omise pour minorer D.

On construit un spline tensoriel P(t,x), C² en espace et temps, compact en x,
avec P(H,x)=0. Des nœuds avant s permettent P(s,x) non nul. Pour un epsilon>=0,
on pose V(t,x)=P(t,x)-epsilon*(H-t), sur [s,H]. Il suffit d'exiger

    V_t + (a/2)*x² V_xx + φ(t,x)*x² >= 0

pour a=amin et a=amax. L'expression étant affine en a, cela couvre tous les
coefficients intermédiaires, même adaptés. La condition globale est vérifiée
par les coefficients exacts de Bernstein sur chaque cellule. Le noyau
polynomial est coupé à ses frontières et centres ; aucune différence de grille
n'est utilisée comme estimation d'erreur.

Pour n'importe quelle proposition numérique P, on peut choisir exactement

    epsilon = max(0, maximum des défauts négatifs de Bernstein).

Cette compensation fait partie de la fonction test ; elle peut être supérieure
à la seule erreur flottante. Hors du support spatial de P, le générateur de V
vaut epsilon, donc la condition reste vraie sans fermer ni absorber les queues.
V et son intégrande stochastique sont bornés sur l'horizon considéré ; Itô et
l'arrêt en H donnent

    D >= E[V(s,Xs)].

On cherche simultanément un portefeuille statique

    g(x) = b0 + b1*x + sum_i lambda_i*(x-Ki)+ <= V(s,x), pour tout x>=0.

Les strikes observés sont des nœuds de partition. Chaque call est affine sur
une cellule, ce qui rend l'inégalité statique vérifiable par Bernstein.
On contrôle aussi x=0 et la pente de la queue droite. La queue n'est pas
supprimée. Une pente positive due au flottant est réparée en diminuant b1 ;
le plus grand défaut statique exact est réparé en diminuant b0. Ces deux
opérations diminuent g partout sur x>=0.

Puis E[Xs]=1 et les intervalles de cotations donnent

    D >= b0+b1 + sum_{lambda_i>=0} lambda_i*bid_i
                   + sum_{lambda_i<0} lambda_i*ask_i.

Le C++ de HiGHS propose P et le portefeuille. Le certificat final n'utilise
aucun statut d'optimalité du solveur : toutes les inégalités, corrections et
sommes sont recalculées en fractions exactes. `denominator_floor.json` conserve
les coefficients, le portefeuille et la preuve brownienne de secours.

Dans le cas initial fourni, le certificat obtenu lors de la validation donne
une borne rationnelle d'environ 0,00013153436 ; sa compression dirigée en
puissance de deux donne le plancher utilisable D>=1/8192. Les nombres précis
peuvent dépendre de la proposition numérique ; leur validité est revérifiée.

## Un certificat LP qui reste effectivement positif

Le plancher delta est choisi dyadique, puis la ligne -sum Z <= -delta est
ajoutée à la relaxation d'occupation. Sa validité physique est justifiée par
le certificat continu ci-dessus, recalculé par le vérificateur.

Tous les coefficients non nuls de D sont 1. Les coefficients -1 et le second
membre dyadique sont exactement représentables en binaire. Le multiplicateur
dual qui vaut 1 sur cette seule ligne et 0 ailleurs donne donc **exactement
delta**, avec résidu de stationnarité nul. La meilleure borne entre ce dual et
le dual numérique proposé est conservée. Le dual numérique brut reste enregistré
et est lui aussi revérifié.

Cela prouve la positivité du minorant enregistré, plutôt que d'espérer que le
résidu du solveur change de signe. Charnes–Cooper peut ensuite utiliser ce
minorant. Si son échelle serait trop mal conditionnée, un statut spécifique
conserve les priors de ratio et la preuve de positivité ; aucune fausse
précision sur N/D n'est annoncée.

## Résoudre une extrémité globale avec une tolérance

Les bornes extérieures L et U portent sur toute la classe continue. Un modèle
admissible dont A est dans [l,u] est à distance au plus U-l du supremum global.
Le modèle minimum est à distance au plus u-L de l'infimum global. Ce sont ces
écarts, et non la convergence du LP de splines, qui autorisent la certification.

`certify` raffine les nœuds sur le support du noyau, réutilise les modèles par
insertion exacte de nœuds et peut résoudre le LP fractionnaire complet ou
générer les contraintes violées avec une base simplex persistante. Les grands
problèmes utilisent aussi un changement diagonal de variables fondé sur les
rayons admissibles des coefficients ; la proposition est toujours ramenée et
revérifiée dans les coefficients physiques rationnels.

Le plafond du problème numérique de proposition et ses bornes supplémentaires
ne définissent jamais la classe extérieure. Ils peuvent limiter la qualité
du modèle trouvé. La meilleure preuve de chaque niveau reste conservée.
`--expand-marginals` permet d'étendre le support temporel des corrections
au-delà des maturités voisines ; tous les prix doivent alors être explicitement
contenus dans leurs spreads. Ce mode peut être plus coûteux.

Une fermeture à 0,01 signifie que les deux modèles sont à moins de 0,01 des
véritables extrémités. Elle n'identifie pas la valeur exacte du supremum quand
un écart positif subsiste. Le budget et les familles finies ne garantissent
pas une fermeture à toute précision pour toute chaîne d'options.

## Une région entière de frontière exactement résolue

Supposons que les cotations contiennent les prix d'un Black de variance a0,
avec amin<a0<amax. Prenons deux dates consécutives sans observation intérieure,
de distance L. Un support temporel de largeur w, strictement intérieur à ce
trou, satisfait la condition suffisante

    w < L * min(a0-amin, amax-a0)/(amax-amin).

Alors, pour **tout centre et toute largeur spatiale K-hK>0**, les deux
extrémités globales de A sont exactement amin et amax, et sont atteintes par
deux variances déterministes C² compatibles avec toutes les observations.

Construction : a est constante égale à la cible sur le support temporel ;
elle est compensée à l'extérieur par une constante q, avec des rampes
quintiques de fraction rho de chaque intervalle extérieur. En notant R=L-w,
l'équation d'horloge est

    a0*L = a0*rho*R/2 + cible*(w+rho*R/2) + q*(1-rho)*R.

La condition stricte ci-dessus permet de choisir rho rationnel strictement
positif assez petit pour que q reste entre les priors pour les deux cibles.
Le code choisit rho par les inégalités algébriques exactes, sans nombre fixe
d'itérations qui échouerait près de la frontière stricte. Les contrôles de
Bernstein des rampes sont leurs deux valeurs aux extrémités, et les raccords
C² sont vérifiés exactement. L'intégrale de variance est a0*T à chaque
maturité observée : les prix sont donc exactement ceux du même Black.
Toute valeur intermédiaire est également atteinte : une combinaison convexe
des deux profils conserve les priors, les raccords C² et l'horloge à chaque
date observée, et son plateau est la combinaison convexe des deux cibles.
L'intervalle annoncé est donc entièrement réalisable sur cette région.

Le même raisonnement vaut pour le trou entre 0 et la première maturité. Tout
support strictement après la dernière maturité permet aussi les deux priors,
avec des rampes placées après les dernières observations.

Cette région est un **résultat sur un continuum de noyaux**, pas une simple
grille. Elle est enregistrée dans `full_prior_region.json`. Elle est
suffisante ; elle n'est pas annoncée comme caractérisation nécessaire pour
les modèles dépendant de l'état. Elle suppose la compatibilité avec le Black
de référence et aucun contrôle quantitatif supplémentaire sur les dérivées
de a. Une hypothèse de pente maximale changerait ce résultat.

Avec les priors [0,0025;0,25], a0=0,04 et le trou [0,25;0,75], la condition est
2*hT<5/66, donc hT<5/132. Sur cette région, l'ensemble d'identification complet
est [0,0025;0,25] et l'erreur minimax absolue vaut 99/800=0,12375. Résoudre cette
frontière signifie prouver une absence d'information supplémentaire dans les
cotations, et non produire une estimation précise de a.

La fenêtre initiale hT=1/16 ne satisfait pas cette condition. Son maximum doit
être traité par les autres bornes et modèles ; le théorème n'y est pas étendu
sans preuve.

## Carte de frontière et limites des annonces

`certified-frontier` exécute les certificats pour chaque combinaison de
T,K,hT,hK explicitement demandée. `requested_grid_certified` n'est vrai que si
toutes les lignes sont fermées à la tolérance. Les fenêtres encore ouvertes
sont conservées. `verify-frontier` vérifie chaque certificat et sa liaison avec
la même classe, les mêmes prix et la même tolérance.

La fermeture d'une grille finie ne prouve pas une frontière universelle sur
tous les noyaux. `universal_frontier_proved` reste faux. Le certificat des
bandes temporelles décrit séparément le continuum où un résultat exact existe.
L'initialisation par mélange Black compatible reste une limitation des modèles
par splines dans cette version ; ce moteur ne certifie pas automatiquement
tous les smiles réels ni les ensembles de désaméricanisation.

## Références et base de confiance

- [Dambis–Dubins–Schwarz, notes de Fabrice Baudoin](https://fabricebaudoin.blog/2012/09/27/lecture-23-time-changed-martingales-and-planar-brownian-motion/).
- [Théorème et preuve, cours de probabilités de Stanford](https://web.stanford.edu/~lindrew/18.676.pdf).
- [Guo, Loeper et Wang, formulation convexe de calibration locale](https://arxiv.org/abs/1709.08075).
- [Documentation HiGHS, interface Python publique](https://ergo-code.github.io/HiGHS/stable/interfaces/python/).

Les arguments de changement de temps, fonctions propres, Itô et couverture
statique sont classiques. Cette livraison ne revendique pas leur nouveauté
bibliographique. Les assemblages rationnels et les séries encadrant les
transcendantes restent la base de confiance ; aucune formalisation dans un
assistant de preuve n'est annoncée.
