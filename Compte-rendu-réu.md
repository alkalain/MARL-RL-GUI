

# 09/12/2025
## Sujet
- Organisation des réunions
- Outillage latex ; gantt 
- Finir le sujet

# 16/01/2026
## Revu du livrable 1
  - Plusieurs modifications à apporter au cahier des charge
  - clarification au sujet du plan de développement


# 29/01/2026
## Points à aborder :
- Bilan sur l'oral
- Clarifier quelle lib a choisir intégrant OvercookedAI
- Notre prochiane étape : mise a niveau sur les librairie 
- l'orga du git

## reponse
- FAIRE LA MISE A NIVEAU AVEC LIB 
- Repasser sur le site de Julien Soulé pour s'inspirer de la structure du code (overcookedai.py)
- integration overcooked : MARLlib (https://marllib.readthedocs.io/en/latest/)
- Diagrame de Classe 
- FAIRE LE KANBAN
- Commencer L'implémentation de l'api
- Commencer la doc : avec un file rouge vulgarisé des différentes étapes. un sorte de perso qui joue à un jeu (l'environment) avec une certaine méthode d'apprentissage (l'algo)

## autre
Eventuel platforme pour avoir plus de puissance de calcule (https://gricad-doc.univ-grenoble-alpes.fr/services/)

# 11/03/2026
## Points à aborder :
- Persentation d'une première version de l'implémentation
## reponse 
- Revoir comment l'exemple de MARLlib fonctionne
- Essayer de lexecuter

# 26/03/2026
## Point aborder 
- Mini projet MARLlib (exemple)
- Installer toute les dépendance necssaire pour marllib
- Fair fonctionner le code de l'exemple et voir les resultats

# 02/04/2026 (par ecrit)
## Point du jour
- L'exemple s'execute correctement
- que faire du fichier progresse.csv, est-ce le fichier pour les résultats
## reponse
- progress.csv est un fichier des statistiques du models
- Le resultat cité au par avant est le rendu visuelle de l'entrainement : voir les blob prédateur chasser les blobs proie.
- Utiliser MLP pour l'architecture et simple world comm pour l'environement avec MPE en hyper paramètre
## objectif
- Manipuler le code pour lancer l'entrainement, sauvegarder des checkpoints puis l'autre code pour charger un checkpoint et avoir un rendu visuel
# 13/04/2026 (par ecrit)
## point aborder 
- tantative reussi avec simple spread
## réponse 
- D'accord, utiliser le sénario simple world comm, simple spread c'est pas le bon scénario

# 04/05/2026 (par ecrit) 
## Point abordé
- le test fonctionne avec entrainement, sauvegarde des checkpoint, test et render
- quel serait la prochaine étape ? on reprend l'intégration avec Mario ?
## reponse 
- Assurez vous que le code fonction de manière stable pour l'exemple
- Si c'est bon passez à l'embalage avec Mario
# 11/05/2026
- Point sur l'avancement du code
- Point sur l'organisation
# reponse
- C'est bien continuer comme ça pour l'organisation
- pour le code aussi
- ajouter l'option de génerer une vidéo ou un gif pour le rendu
- prochaine étape faire l'intégration des hpo
- Fair une MVP du GUI est concevable
