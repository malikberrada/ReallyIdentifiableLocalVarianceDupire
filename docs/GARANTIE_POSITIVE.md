# Garanties positives sur des moyennes de variance locale

Le programme part des spreads de calls européens normalisés. Dans les exemples, le prix initial vaut 1 et la variance instantanée annualisée \(a\) appartient à \([1/400,1/4]\). Il calcule un estimateur et un majorant de son erreur absolue valable pour chaque diffusion compatible avec toutes les cotations. Un modèle témoin vérifié assure que cette classe contient au moins une diffusion.

## Résultats reproductibles

| Cotations | Poids et domaine | Estimation affichée | Erreur absolue au plus |
| --- | --- | ---: | ---: |
| R3, 27 calls | Gamma du call de strike 1, de 0 à 9 mois, prix normalisé compris entre 0,65 et 1,50 | 0,039810 | 0,009797 |
| R3, 27 calls | Même corridor, gamma du call de strike 1, de 0 à 1 an | 0,039420 | 0,008310 |
| R3, 27 calls | Fenêtres bêta de la sous-famille `hjb-early` de `2/r0000` | 0,0551 | 0,0526 |
| `general_informative`, 14 calls synthétiques à spreads étroits | Famille bêta entière, calcul direct depuis le CSV | 0,2499029702 | 0,0000970298 |
| Mêmes 14 calls | Même famille, relecture du certificat `solve-general` livré avec la version 0.5.1 | 0,2461328125 | 0,0038671875 |

Pour les deux premières lignes, la variance de référence \(b=0,04\) définit le gamma. Les diffusions admises peuvent avoir une variance qui change avec la date et le prix. Le corridor porte sur le prix du sous-jacent à chaque instant, dans les unités du CSV. L'erreur s'exprime en unités absolues de variance annualisée. Une estimation ponctuelle de \(a(t,s)\) et une couverture à 100 % des quatre régions bêta R3 sont des objectifs distincts. Le rejeu global fourni dans la campagne conserve le statut `INCOMPLETE`.

Les coordonnées `left` de `hjb-early` sont les bords gauches \(u\in[0,25\,;0,251]\), \(v\in[0,25\,;0,252]\) et les demi-largeurs \(h_t\in[0,25\,;0,251]\), \(h_k\in[0,749\,;0,751]\). Dans `general_informative`, les centres vérifient \(t\in[0,49\,;0,51]\) et \(k\in[0,99\,;1,01]\), les demi-largeurs \(h_t\in[0,23\,;0,24]\) et \(h_k\in[0,64\,;0,65]\).

Le pilote `python scripts/reproduce.py` régénère directement les deux chaînes synthétiques et les deux certificats gamma. Les commandes individuelles ci-dessous utilisent l’arborescence d’exemples restaurée par `scripts/install_supplement.py`.

## Commandes PowerShell

Depuis la racine du dépôt, utiliser l’environnement du guide principal. Les commandes ci-dessous utilisent PowerShell.

```powershell
$py = ".\.venv\Scripts\python.exe"
$gp = ".\scripts\garantie_positive.py"
$ex = ".\examples"
$qR3 = "$ex\exact_initial\normalized.csv"
$out = ".\runs\garantie-positive"
```

Ces commandes lisent le même CSV que la campagne R3. Son SHA-256 vaut `aaf81cb73eca2b79049f1c0255853bca4f4c39b6645775d24ed314a5a028b656`.

```powershell
foreach ($id in "synthetic_1_4", "synthetic_2_4") {
    & $py -I $gp gamma --quotes $qR3 --call-id $id --variance-reference 0.04 --corridor 0.65 1.50 --epsilon 0.01 --decimales 6 --out "$out\$id.json"
    if ($LASTEXITCODE -ne 0) { throw "Cible non validée pour $id (code $LASTEXITCODE)." }
}
Get-Content "$out\synthetic_2_4.json" -Raw
```

Les deux certificats de référence sont fournis avec le patch. Les lignes suivantes comparent les fichiers reproduits octet par octet au moyen de leur SHA-256.

```powershell
foreach ($id in "synthetic_1_4", "synthetic_2_4") {
    $ref = (Get-FileHash "$ex\garantie_positive\$id.json" -Algorithm SHA256).Hash
    $calcule = (Get-FileHash "$out\$id.json" -Algorithm SHA256).Hash
    if ($ref -ne $calcule) { throw "Empreinte différente pour $id." }
}
```

Le calcul bêta sur le jeu synthétique à spreads étroits et la relecture de la frontière existante se lancent ainsi.

```powershell
& $py -I $gp cotations --quotes "$ex\general_informative\normalized.csv" --domaine "$ex\general_informative_domain.json" --epsilon 0.0001 --out "$out\beta-informative.json"
& $py -I $gp general --run "$ex\general_informative" --epsilon 0.01 --out "$out\frontiere-informative.json"
```

Pour recalculer la preuve HJB sur la famille précoce de la chaîne R3, utiliser le dossier du rejeu terminé sur sa machine. Le script lie le reçu et les preuves locales à la campagne, recalcule les résidus HJB et la correction de couverture, vérifie la positivité du dénominateur et les 27 prix du témoin.

```powershell
& $py -I $gp r3 --run ".\runs\r3" --workers 10 --epsilon 0.0526 --decimales 4 --out "$out\beta-r3-early.json"
& $py -I .\tests\test_garantie_positive.py -v
```

Le code de sortie 0 valide la précision de l'estimateur rationnel exact. Le code 2 après production du rapport indique un encadrement valide dont le rayon dépasse la cible. Le code 1 signale un échec de vérification. Dans le JSON, `garantie_uniforme.decimal.precision_cible_garantie` porte sur la valeur décimale effectivement affichée. Son erreur est recalculée depuis les deux bornes puis arrondie par excès. Un petit nombre de décimales peut rendre l'affichage trop grossier pour la cible alors que l'estimateur rationnel la satisfait. `intervalle_resserre_par_rapport_au_prior` indique si les bornes de départ ont été resserrées.

## Preuve sur le corridor d'un call

Pour un poids positif \(\phi(t,s)\), la quantité estimée est

$$
A_\phi =
 \frac{\mathbb E\int \phi(t,S_t)\,a_t S_t^2\,dt}
      {\mathbb E\int \phi(t,S_t)\, S_t^2\,dt}.
$$

Dans la normalisation martingale \(dS_t=S_t\sqrt{a_t}\,dW_t\), soit \(\Gamma_b\) le gamma du call Black de maturité \(T\), de strike \(K\), de variance de référence \(b\) et de prix initial \(B_b\). La formule d'Itô pour son prix de couverture donne

$$
2(C-B_b)=\mathbb E\int_0^T (a_t-b) S_t^2\Gamma_b(t,S_t)\,dt .
$$

Le prix \(C\) du modèle appartient au spread observé. Pour le corridor \(J\), poser \(\phi(t,s)=\Gamma_b(t,s)\,\mathbf1_{\{s\in J\}}\) sur \([0,T)\). Le calcul certifie un minorant \(d\) de \(D_\phi=\mathbb E\int\phi S_t^2dt\) et un majorant \(H\) de \(\mathbb E\int_0^T \mathbf1_{\{S_t\notin J\}}S_t^2\Gamma_bdt\). On obtient

$$
 2(\mathrm{bid}-B_b)-(a_{\max}-b)_+ H
 \ \le\ (A_\phi-b)D_\phi\
 \le\ 2(\mathrm{ask}-B_b)+(b-a_{\min})_+ H .
$$

Lorsque \(d>0\), on divise la partie négative de la borne gauche (au sens de son minimum avec zéro) et la partie positive de la borne droite par \(d\), puis on ajoute \(b\) et on intersecte avec le domaine de variance. Dans les exemples à neuf mois et à un an, les dénominateurs sont respectivement supérieurs à 0,41998 et 0,53495. Les masses hors corridor sont respectivement inférieures à 0,000925 et 0,003597.

Le calcul de \(d\) utilise 24 tranches de temps et 48 bandes de prix emboîtées dans le corridor. Pour \(t\) antérieur à une échéance cotée, la convexité et la propriété de martingale majorent l'espérance du payoff d'un call ou d'un put à \(t\) par son ask à cette échéance. La parité call-put donne l'ask du put. Sur une bande \([l,r]\), un put de strike \(k>l\) majore \(\Pr(S_t<l)\) par \(\mathrm{ask}_{put}/(k-l)\). Un call de strike \(k<r\) majore \(\Pr(S_t>r)\) par \(\mathrm{ask}_{call}/(r-k)\). Les différences de minorants successifs de \(s^2\Gamma_b\) sur les bandes donnent un minorant de son espérance par somme d'indicatrices. La dernière tranche contribue zéro à ce minorant.

Pour \(v=b(T-t)>0\), le contrôle des queues utilise l'identité

$$
s^2\Gamma_b(t,s)
 = \frac{\sqrt{sK}}{\sqrt{2\pi v}}
   \exp\!\left(-\frac{\log^2(s/K)}{2v}-\frac{v}{8}\right).
$$

Sur chaque queue, on majore ce poids et la probabilité de présence. La borne globale \(K/\sqrt{2\pi v}\) est intégrable jusqu'à \(T\). Omettre `--corridor` utilise directement \(\Gamma_b\) sur tous les prix avec \(H=0\).

## Preuve pour une famille de fenêtres bêta

Pour la commande `cotations`, le poids est le produit des deux bosses bêta du projet. En choisissant \(b=a_{\max}\), l'intégrande \((b-a_t)S_t^2\Gamma_b\) est positif. Le code construit des coefficients rationnels positifs tels que \(\phi\le\sum_i\lambda_i\Gamma_{b,i}\) partout sur le support de chaque fenêtre. Il en déduit \(A_\phi\ge b-2\sum_i\lambda_i(B_{b,i}-\mathrm{bid}_i)/d\). Le choix \(b=a_{\min}\) donne de même une borne supérieure avec les asks. Un minorant commun \(d>0\) du dénominateur découle de \(\mathbb E[(S_t-1)^2]\le e^{a_{\max}t}-1\) et d'une bande de prix où toutes les bosses ont une valeur positive.

Les logarithmes, exponentielles, prix Black, gammas et arrondis de certification utilisent les bornes rationnelles du dépôt. Les commandes `gamma` et `cotations` recherchent automatiquement un témoin de variance constante, puis vérifient ses prix sur toute la chaîne. Pour une autre chaîne, on peut spécifier `--variance-temoin` ou le dossier certifié d'un modèle plus général avec `--modele-temoin`. La recherche d'un témoin peut échouer et la précision obtenue dépend des spreads et du domaine considéré.

La méthode de couverture s'appuie sur la formule classique d'erreur de couverture de Black. [El Karoui, Jeanblanc-Picqué et Shreve, 1998](https://doi.org/10.1111/1467-9965.00047)
