# Reproduire la campagne R3 du 18 novembre 2026

Depuis la racine du dépôt, avec Python 3.12 et les dépendances de `requirements/review.txt`, exécuter les commandes suivantes sous PowerShell ou Linux.

```console
python scripts/install_supplement.py DupireIdentify-supplement-20261118.zip
python -I scripts/rejouer_r3.py tout --workers 4 --out runs/r3
python -I scripts/rejouer_r3.py comparer --out runs/r3
```

Le supplément numérique accompagne la soumission. L’installateur vérifie sa taille et son SHA-256 avant de déposer la campagne d’origine dans `rejeu_r3/campagne.zip`. Le lanceur vérifie à nouveau le SHA-256 de cette campagne. Le rejeu scientifique fonctionne ensuite hors ligne. Pour ne recalculer qu’une étape, les actions `optimiser` et `simuler` acceptent les mêmes arguments.

## Origine des résultats

| Étape | Calcul effectué | Entrées archivées |
|---|---|---|
| Optimisation des flux | HiGHS reconstruit 16 des 77 flux d’horloge avec leurs masses entières exactes. Pour les 77, le programme recalcule le poids minimal du modèle de référence compatible avec les 27 cotations. | Les fenêtres et paramètres de rejeu sont dans `horloges.json`. Les 61 autres flux repartent des décisions enregistrées. |
| Optimisation des portefeuilles | Résolution de 1 940 programmes linéaires calendaires à partir des fenêtres et des prix. Recalcul des portefeuilles complémentaires, des couvertures HJB et des trois certificats HJB avec génération des portefeuilles de résidu. | Grilles, familles de fenêtres, trois surfaces HJB et multiplicateurs de la campagne. |
| Simulation | Propagation des masses des 60 politiques gaussiennes, recalcul des prix et des intégrales signées. Relecture des preuves sur les horloges et les surfaces. | Les 60 tableaux de décision entiers sont fournis. La propagation est déterministe et ne demande aucune graine aléatoire. |
| Audit | Relecture de 34 350 certificats et de leur complément de 29 957 certificats, reconstruction des volumes sur les quatre régions et des encadrements aux quatre points publiés. | Pour 540 entrées plus anciennes, la campagne réutilise les registres et reçus historiques authentifiés par SHA-256. |

Les fenêtres de `horloges.json` sont des recettes vérifiées qui redonnent les témoins correspondants. Pour les autres flux et les politiques gaussiennes, la trace disponible commence aux décisions sélectionnées. Le détail des essais exploratoires reste inconnu. La recherche des portefeuilles HJB repart des surfaces numériques conservées. Pour consulter les routines mathématiques d’origine, ouvrir `campagne.zip`, notamment `search_flows.py`, `gaussian_chain.py`, `verify.py` et `COMPLEMENT.md`.

Pour les 16 flux régénérables, le lanceur calcule d’abord les coefficients numériques à partir des fenêtres et des cotations. Une différence d’arrondi entre processeurs peut déplacer une unité de masse. Dans ce cas, HiGHS résout à nouveau les sous-problèmes avec les coefficients binary64 archivés dans `coefficients_flux.npz`. Ce fichier conserve les contraintes de prix et les objectifs des 39 sous-problèmes. Le solveur calcule leurs solutions sur la machine qui exécute le rejeu.

Après `optimiser`, `simuler` contrôle les empreintes des entrées régénérées et les utilise. Pour rejouer directement la simulation à partir des décisions archivées, lancer `simuler` avec un nouveau dossier `--out` qui ne contient pas d’optimisation.

## Résultats attendus

| Région R3 | Couverture du minimum | Couverture du maximum | Couverture conjointe |
|---|---:|---:|---:|
| 1/r0000 | 100 % | 90,0450 % | 90,0450 % |
| 1/r0001 | 100 % | 95,9124 % | 95,9124 % |
| 2/r0000 | 100 % | 87,6048 % | 87,6048 % |
| 2/r0001 | 100 % | 92,3993 % | 92,3993 % |

Ces pourcentages mesurent des fractions du volume des fenêtres. Les fractions rationnelles exactes figurent dans `simulation/coverage.json`. Son SHA-256 attendu est `aeb0b0d27fb0ecefc3e46398418015b8cb1b21ca7ba18c0d90a6632e254118b7`.

| Fichier dans le dossier de sortie | Lecture |
|---|---|
| `optimisation/optimisation.json` | Nombre de programmes, de flux et de certificats HJB régénérés. Les coefficients recalculés se trouvent dans le même dossier. |
| `simulation/coverage.json`, `simulation/completion.json` | Fractions exactes et critère de clôture des quatre régions. Le bilan publié donne `request_completed = false`. |
| `simulation/points.json` | Encadrements rationnels de chacun des deux extrêmes aux quatre fenêtres publiées. |
| `simulation/open-*.json` | Sous-domaines qui restent ouverts dans chaque région. |
| `simulation/proof-replay.json`, `simulation/supplement-replay.json` | Reçus, prix et empreintes des témoins. |
| `simulation/reproduction.json` | Comparaison des neuf fichiers avec les neuf originaux, octet par octet via SHA-256. Le rejeu réussi donne `reproduction_identique = true`. |

Le vérificateur scientifique renvoie 2 pour la couverture partielle publiée. Le lanceur renvoie 0 après reproduction exacte, y compris pour cet état partiel. Une différence de données, de prix, de certificat ou de fichier fait échouer le rejeu et produit un code non nul.
