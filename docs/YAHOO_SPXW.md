# Certifier une moyenne gamma sur une chaîne Yahoo SPXW

Depuis la racine du dépôt, le guide principal installe les versions de référence. Les calculs portent sur une échéance et sur tous les calls et puts retenus à cette échéance. [Installation et résultats](../README.md)

## Rejouer la validation historique

Pour cette expérience antérieure, placer sa copie locale de `raw_snapshot.json` dans `examples/yahoo_spxw/`. Les empreintes attendues figurent dans `manifest.json`.

```powershell
$py = ".\.venv\Scripts\python.exe"
$yf = ".\scripts\yahoo_spxw.py"
$ex = ".\examples\yahoo_spxw"
& $py -I $yf verify --run $ex
$tag = Get-Date -Format "yyyyMMdd_HHmmssfff"
$out = ".\runs\spxw_rejeu_$tag"
& $py -I $yf certify --snapshot "$ex\raw_snapshot.json" --calendar "$ex\session_closes.json" `
    --expiry 2026-11-09 --variance-reference 0.04 `
    --corridor 0.65 1.50 --epsilon 0.01 --decimales 6 --out $out
& $py -I $yf verify --run $out
```

`verify` recalcule les intervalles de prix du témoin dans les spreads bruts, les sommets de parité, les portefeuilles qui minorent le dénominateur, le contrôle des queues et l’encadrement final. Il utilise les paramètres du témoin et les positions archivés. Une nouvelle recherche avec d’autres versions de SciPy ou HiGHS peut proposer des positions différentes. Chaque rejeu d’un même dossier exige l’égalité de la preuve recalculée.

## Exemple livré

Les octets de `raw_snapshot.json` sont ceux de `input/coherent_yahoo_20261026T193126_940753Z/attempt_01/raw_snapshot.json` dans `audit_yahoo_20261026.zip`. Leur SHA-256 est `0c44b9b66a46b238d891ffbeb21fcaf65299f4987a9cf8b9f51993921e60db41`. L’acquisition de six échéances commence le 26 octobre 2026 à 19:31:29,106182 UTC et finit à 19:31:43,546751 UTC. L’instant de valorisation conventionnel est le début de la collecte. Le sous-jacent de la tranche analysée porte l’horodatage 19:16:27 UTC et indique un délai de 15 minutes. Yahoo fournit `lastTradeDate` pour la transaction la plus récente du contrat. La simultanéité des fourchettes demeure une hypothèse.

Pour l’échéance du 9 novembre, la collecte contient 255 lignes, dont 78 strikes avec une paire exploitable. La règle par défaut prend la paire la plus proche de chacun des niveaux 98 %, 99 %, 100 %, 101 % et 102 % du spot brut. Elle sélectionne 7 525, 7 605, 7 680, 7 755 et 7 830. Le spot sert uniquement à sélectionner les strikes. L’algorithme ne retire aucune paire après calibration. `audit.json` donne une raison pour chaque ligne écartée dans la tranche, la liste des autres échéances et les heures des transactions retenues.

La variance locale annuelle a priori est dans `[0.0025, 0.25]`. Le corridor du prix normalisé est `[0.65, 1.50]`. La variance de référence du premier noyau est fixé à 0.04. Le polygone de parité autorise `1487/1500 ≤ D ≤ 377/375`. La projection du forward va exactement de `5759445/749` à `1145850/149`. Le témoin vérifié utilise `D = 599/600` et un forward d’environ 7 689,9854. Ses dix intervalles de prix sont dans les bid–ask. Leur plus petite marge intérieure dépasse 0,0400008 point d’indice pour le témoin livré.

| Référence gamma du call de strike 7 680 | Estimation à six décimales | Erreur absolue garantie | Intervalle extérieur arrondi vers l’extérieur |
| --- | ---: | ---: | --- |
| 0,04 | 0,018988 | 0,009501 | [0,009487, 0,028488] |
| `auto`, environ 0,019931848 | 0,019929 | 0,000662 | [0,019267, 0,020590] |

La première ligne est fournie dans `certificate.json`. Pour reproduire la seconde, relancer `certify` avec `--variance-reference auto` et un nouveau `--out`. Chaque référence définit sa propre moyenne gamma. Pour la référence 0,04, le dénominateur vaut au moins 0,307838 et la masse gamma hors corridor vaut au plus `6e-23`. Les fractions complètes sont conservées. L’unité de la moyenne et de l’erreur est la variance annuelle.

Les 78 paires prises ensemble imposent `D ≥ 53/50` par les strikes 7 770 et 7 775, et `D ≤ 49/50` par les strikes 7 775 et 7 790. Pour reproduire la preuve d’incompatibilité, exécuter les deux commandes suivantes. La première rend le code 1 et écrit `infeasibility.json`. La seconde vérifie sa preuve et rend 0.

```powershell
$full = ".\runs\spxw_toutes_$tag"
& $py -I $yf certify --snapshot "$ex\raw_snapshot.json" --calendar "$ex\session_closes.json" `
    --expiry 2026-11-09 --all-pairs --out $full
& $py -I $yf verify --run $full
```

## Collecter une nouvelle chaîne

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmssfff'
$snap = ".\runs\spxw_brut_$tag"
$out = ".\runs\spxw_gamma_$tag"
& $py -I $yf collect --min-days 7 --max-days 365 --max-expiries 3 --out $snap
if ($LASTEXITCODE -ne 0) { throw "Collecte incomplète, voir $snap\raw_snapshot.json" }
& $py -I $yf certify --snapshot "$snap\raw_snapshot.json" --variance-reference 0.04 `
    --corridor 0.65 1.50 --epsilon 0.01 --out $out
if (Test-Path "$out\manifest.json") { & $py -I $yf verify --run $out }
```

`collect` archive les tableaux complets reçus avant filtrage, la réponse sous-jacent par échéance, les heures UTC des requêtes, les erreurs et la version de yfinance. Il sélectionne les premières échéances de `^SPX` dans la fenêtre de dates civiles de New York demandée. L’accès utilise `Ticker.options` et `Ticker.option_chain(expiry)`. Chaque requête a un délai maximal et un nombre borné de nouvelles tentatives. Une réponse partielle reste disponible sur disque. Le collecteur s’ouvre par défaut pendant la séance normale du marché au comptant. `--allow-closed` archive explicitement un relevé hors séance. Yahoo peut retourner une liste vide, des carnets nuls ou une erreur de débit.

`certify` traite par défaut la première échéance `^SPX` archivée. `--expiry YYYY-MM-DD` la fixe. On peut imposer des strikes avec `--strikes 7525 7605 7680 7755 7830`, modifier les cinq niveaux par `--anchors`, choisir `--all-pairs`, fixer `--call-id`, demander `--full-space` ou indiquer un intervalle externe justifié avec `--discount D_MIN D_MAX`. Chaque paramètre fait partie de la preuve. Deux strikes distincts au minimum sont nécessaires. Des niveaux qui sélectionnent la même paire entraînent un diagnostic. Pour une collecte historique dans l’ancien format `yfinance` brut, `--calendar` peut fournir une table `source, sessions` au même format que l’exemple. Sa provenance et ses octets sont conservés. Pour les nouveaux relevés, les clôtures du calendrier XNYS sont gelées à la collecte, y compris celles des demi-séances. La maturité est la différence UTC exacte en ACT/365 fixe.

| Sortie | Interprétation |
| --- | --- |
| `certificate.json`, code 0 de `certify` | Un témoin et un intervalle extérieur sont vérifiés. Le rayon exact et l’erreur affichée satisfont la cible. |
| `certificate.json`, code 2 de `certify` | La preuve de l’intervalle est disponible et l’erreur demandée reste dépassée. |
| `infeasibility.json`, code 1 de `certify` | Deux inégalités rationnelles de parité sont en contradiction sur les livres retenus. |
| `failure.json`, code 1 de `certify` | Le motif est enregistré. L’échec de la recherche de témoin laisse ouverte la question de la faisabilité générale. |
| Code 0 de `verify` | La preuve archivée, positive ou d’incompatibilité, a été recalculée. |

## Construction mathématique

**Ensemble des courbes.** Pour chaque strike brut \(K_i\), poser \(A=DF\), \(l_i=C_i^b-P_i^a\), \(u_i=C_i^a-P_i^b\). La parité impose \(l_i\le A-DK_i\le u_i\). La positivité du sous-jacent donne \(A\ge\max_i C_i^b\) et \(D\ge\max_i P_i^b/K_i\). Deux strikes distincts bornent \(D\) par différence. L’intersection des demi-plans est calculée en fractions, y compris quand elle se réduit à un segment ou un point. Les extrêmes de \(KD/A\), \(A/D\) et des prix divisés par \(A\) se déduisent des sommets par positivité des dénominateurs. Les calls et les puts resserrent conjointement les fourchettes normalisées. Huit tranches en \(D\) permettent de conserver davantage de dépendance pour le résidu de prix du call. Une valeur du couple \((D,F)\) choisie pour construire un témoin n’intervient jamais comme une hypothèse de courbe unique dans l’intervalle extérieur.

**Témoin continu.** La recherche essaie d’abord une variance constante. La famille qui passe sur l’exemple utilise \(Z=W_T/\sqrt T\), \(S_T=B\exp h(Z)\), où \(h\) est continue, affine par morceaux, de pentes \(s_j>0\). Poser \(F=\mathbb E[S_T]\) et \(X_t=\mathbb E[S_T\mid W_t]/F\). C’est une martingale positive issue de 1. Pour \(t<T\), sa volatilité relative est

\[
\frac{\mathbb E[\exp h(Z)\,h'(Z)\mid W_t]}
     {\sqrt T\,\mathbb E[\exp h(Z)\mid W_t]}.
\]

C’est une moyenne pondérée de \(s_j/\sqrt T\). L’application \(W_t\mapsto X_t\) est strictement croissante, donc cette volatilité est une fonction déterministe de \((t,X_t)\). Vérifier \(a_{\min}\le s_j^2/T\le a_{\max}\) pour chaque pente borne la variance du processus entier. Un coefficient constant admissible prolonge le processus après \(T\). Des courbes déterministes positives reliant le spot à \(F\) et le facteur initial 1 à \(D\) donnent le prix brut \(D\,\mathbb E[(S_T-K)^+]\) et son put. Seules leurs valeurs à l’échéance sont contraintes ici.

Sur un segment où \(h(z)=b+sz\), l’intégrale de \(\exp h\) contre la normale entre \(l\) et \(r\) vaut \(\exp(b+s^2/2)[\Phi(r-s)-\Phi(l-s)]\). Le seuil d’exercice s’obtient en intersectant le segment avec \(z>(\log(K/B)-b)/s\). Les fonctions du module `interval.py` encadrent ces formules par des fractions. L’optimisation flottante propose un seul vecteur de paramètres. Le vérificateur recalcule l’intégralité des prix en points d’indice et exige que chaque intervalle soit inclus dans le spread original. La variance de référence du noyau gamma et les pentes du témoin sont des paramètres distincts.

**Observable.** Pour chaque courbe admissible le même contrat coté a le strike normalisé \(k=K/F\). Écrire \(\Gamma_r(t,x;k,T)\) pour le gamma Black normalisé à variance annuelle de référence \(r\). L’intervalle couvre uniformément

\[
V=\frac{\mathbb E\int_0^T a(t,X_t)X_t^2\Gamma_r(t,X_t;k,T)\mathbf1_{L\le X_t\le U}\,dt}
        {\mathbb E\int_0^T X_t^2\Gamma_r(t,X_t;k,T)\mathbf1_{L\le X_t\le U}\,dt}.
\]

Le gamma porte sur le vrai \(K/F\) de chaque courbe compatible. La formule d’Itô donne \(2(c-B_r)=\mathbb E\int (a-r) X_t^2\Gamma_r\,dt\) sur tout l’espace, où \(c=C/A\). L’identité s’obtient jusqu’à \(T\) par approximation du payoff convexe et passage à la limite. Les bornes a priori de variance et l’estimation intégrable \(X^2\Gamma_r\le k/\sqrt{2\pi r(T-t)}\) contrôlent le terme intégral. On encadre le résidu avec les prix de calls et de puts, on soustrait la contribution extérieure, puis on divise par l’intervalle strictement positif du dénominateur dans le corridor.

**Minorant du dénominateur.** Pour \(v=r(T-t)\), on utilise

\[
x^2\Gamma_r=\frac{\sqrt{kx}}{\sqrt{2\pi v}}\exp\!\left(-\frac{\log^2(x/k)}{2v}-\frac v8\right).
\]

Sur chaque tranche de temps et chaque cellule de prix, cette formule fournit un minorant uniforme en \(k\). Un portefeuille statique affine en \(x\), diminué de calls et de puts en quantités positives, peut minorer cette fonction tronquée. Les strikes de ces instruments sont fixés à des extrémités conservatrices des strikes normalisés. Leur espérance avant \(T\) est majorée par l’ask normalisé à \(T\), par convexité du payoff et propriété de martingale. Les sommets du maillage comprennent tous les changements de pente du portefeuille. Le vérificateur recalcule sa composante en cash en arithmétique rationnelle à chacun de ces sommets et contrôle sa pente sur la dernière cellule non bornée. Cela valide l’inégalité entre les sommets, pour tout prix positif. Il combine ce minorant avec celui de bandes emboîtées et avec \(\mathbb E[(X_t-1)^2]\le e^{a_{\max}t}-1\). La dernière tranche a un minorant nul et un majorant intégré fini.

**Hors corridor.** Les asks des puts et des calls majorent les probabilités des deux queues par ordre convexe et inégalité de Markov. Sur chaque queue, on majore le gamma pondéré par son maximum global ou par sa valeur maximale au bord quand la position du pic le permet. L’intégrale de \(1/\sqrt{T-t}\) borne aussi la tranche terminale. Les endpoints, les positions, le minorant par tranche et les masses extérieures sont conservés dans `certificate.json`. Le certificat produit un encadrement extérieur et un témoin de non-vacuité. L’optimalité des deux extrêmes et la construction de modèles qui les approchent restent des questions distinctes de P1.

## Contrats américains et P3

La table officielle de Cboe identifie `SPXW` comme la racine des options SPX européennes, réglées en numéraire à la clôture. Les racines AM, les contrats `SPY`, les contrats ajustés, les expirations ou types contradictoires et les identifiants incompatibles avec le strike reçoivent un motif d’exclusion. `SPY` est américain. Le projet contient déjà un moteur P3 de recherche numérique de vecteurs européens par modèle commun. Une garantie de désaméricanisation demanderait un encadrement conjoint de l’arrêt optimal et des sauts aux dates de dividendes. Ce patch valide le cas européen à une échéance. Toute affirmation commune à plusieurs échéances demanderait également un témoin valable simultanément sur ces échéances.

[Cboe, spécifications et tableau comparatif SPXW et SPY](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications/) · [Cboe, horaires d’expiration SPXW](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications) · [NYSE, calendrier du comptant](https://www.nyse.com/trade/hours-calendars) · [yfinance, champs effectivement retournés par `option_chain`](https://github.com/ranaroussi/yfinance/blob/main/yfinance/ticker.py)
