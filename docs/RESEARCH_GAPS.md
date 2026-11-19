# Limites et étapes mathématiques restantes — 0.5.0

La livraison fournit un solveur général de bornes et des témoins vérifiés pour
une chaîne finie donnée. Elle ne résout pas universellement tous les smiles.

- Les LP d'occupation sont des relaxations extérieures. Leur complétude et une
  règle de raffinement garantissant la fermeture aux vrais extrêmes ne sont pas
  démontrées ici. Raffiner peut rester insuffisant.
- Les coefficients intérieurs sont une famille constructive riche, mais la
  densité de cette famille parmi tous les modèles compatibles n'est pas prouvée.
  La calibration est une recherche locale à budget explicite, sans garantie
  d'existence trouvée ni d'optimalité globale.
- Une LP faisable ne prouve pas l'existence d'une diffusion. Une Phase I duale
  positive peut prouver une classe vide; sinon la faisabilité peut rester inconnue.
- Les modèles sont C² en état et constants par morceaux en temps. Des bornes
  prescrites de dérivées temporelles changeraient l'ensemble d'identification.
- La région future est exacte sous existence d'un modèle vérifié. Le seuil de
  petit support utilise une marge stricte dans tous les spreads et constitue
  seulement une condition suffisante, souvent très conservatrice.
- D>0 qualitativement ne signifie pas qu'un minorant numérique exploitable a
  été obtenu. Aucun epsilon artificiel ne remplace une preuve de positivité.
- Les paramètres sont positifs en coordonnées de supports gauches et de
  demi-largeurs. Les fenêtres dont le support traverse t=0 ou k=0 ne sont pas
  définies par ce workflow.
- L'union dénombrable des boîtes couvre l'espace positif; aucun préfixe fini ne
  constitue une preuve sur cette union. Les budgets arithmétiques peuvent aussi
  laisser une boîte ouverte.
- Une méthode d'estimation et une précision de certificat ne réduisent pas à
  elles seules l'intervalle d'identification. Les modèles indiscernables donnent
  une obstruction minimax d'au moins la moitié de leur écart au même noyau.
- Les exemples fournis sont synthétiques. La normalisation réelle, les spreads
  exécutables et la désaméricanisation commune avec dividendes restent à valider
  sur chaque source de marché. P3 n'est pas généralisé par ce patch.

Il manque donc un théorème de convergence constructive et de terminaison,
avec contrôle uniforme des queues et des cas limites, avant de pouvoir annoncer
une frontière universelle atteignable pour toute chaîne admissible.
