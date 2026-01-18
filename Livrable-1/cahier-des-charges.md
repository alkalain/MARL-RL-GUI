---
title: "MARIO: Cahier des charges"
student-name: "Alain KARAPETIAN, Timeo RAPHOZ, Enzo VIGNE"
academic-year: "2025-2026"
master-degree: "Master Mathématiques Informatiques Appliquées aux Sciences Humaines et Sociales (MIASHS)"
master-year: "Master 1"
date: "Janvier 2026"
logo: "logo_uga.png"
coverpage: true
---
# PROJET MARIO

## *Objectif général*

L’objectif du projet est de rendre l’utilisation de techniques de *Reinforcment Learning (RL)* et *Multi-Agent Reinforcment Learning (MARL)* les plus agnostiques de connaissances expertes que possible. Cette plateforme doit s’appuyer sur des technologies (librairies) existantes afin de permettre au mieux d’automatiser des tâches techniques habituellement réalisées à la main.

À noter que  ce projet a pour vocation d’avoir une application locale, ce qui veut dire que son API n’est utilisable que localement pour l’application.  

Ce logiciel doit permettre de choisir un environnement de jeu vidéo, un ou plusieurs algorithme(s) d'apprentissage par renforcement, tester et visualiser le resultat de l'entrainement en entrant des comandes ou interagissant avec une interface graphique. Ces fonctionalité ne sont pas toutes réunies au sein d'une solution / logiciel / plateforme unique. Pour justifier le positionnement de ce projet, il est donc nécessaire de le situer par rapport aux solutions actuelles.


## *Analyse de l’existant*
Le tableau suivant permet de mettre évidence les manques spécifiques que la plateforme MARIO vise à combler, il présente un comparatif entre des platforme existantes et celle de notre projet. 

\begin{center}
\begin{longtable}{|p{5cm}|c|c|c|c|c|}
\hline
\textbf{Critère / Plateformes} & \textbf{Gama} & \textbf{Netlogo} & \textbf{Cromas} & \textbf{Matlab} & \textbf{MARIO} \\ \hline
\endfirsthead
\hline
\textbf{Critère / Plateformes} & \textbf{Gama} & \textbf{Netlogo} & \textbf{Cromas} & \textbf{Matlab} & \textbf{MARIO} \\ \hline
\endhead
Configuration d'agent (RL) & [ ] & [ ] & [ ] & [x] & [x] \\ \hline
Agnostique aux langages & [ ] & [x] & [x] & [ ] & [x] \\ \hline
Documentation & bonne & bonne & moyenne & Moyenne & bonne \\ \hline
Interface de visualisation & [x] & [x] & [x] & [x] & [x] \\ \hline
\end{longtable}
\end{center}

Après avoir compris la plus-value du projet, nous pouvons maintenant détailler les différentes étapes de notre mise en oeuvre.

# MISE EN OEUVRE

Dans le cadre du projet, plusieurs tâches doivent être réalisées. Celles-ci correspondent aux fonctionnalités à implémenter afin d’assurer le bon fonctionnement de la plateforme. Ces tâches peuvent être organisées et planifiées de manière itérative et chronologique. Les différentes parties présentées ci-dessous détaillent l’ensemble des tâches ainsi que leurs critères de validation. Il est toutefois important de préciser que les technologies proposées sont susceptibles d’évoluer si d’autres solutions sont jugées plus appropriées.

## *Revue de littérature et montée en compétences*

Cette première phase du projet est consacrée à la préparation et à la structuration des bases nécessaires à son bon déroulement. Cette phase préliminaire a pour objectif d’assurer une compréhension commune et approfondie du sujet, des technologies, outils et frameworks associés à travers une revue de littérature approfondie et l’étude des concepts fondamentaux liés au Reinforcement Learning (RL) et au Multi-Agent Reinforcement Learning (MARL). Elle permet notamment d’identifier les approches existantes, les bonnes pratiques, les limites des solutions actuelles et les choix technologiques pertinents pour la conception et le développement de la plateforme.

Cette étape est indispensable et constitue un prérequis au lancement des phases de conception et d’implémentation du projet.

## *Développement de l’API*

Une API (Interface de Programmation d’Application) est un ensemble de fonctions et de protocoles permettant à différents logiciels ou composants de communiquer entre eux. Elle sert à exposer les fonctionnalités nécessaires au projet de manière standardisée, pour que d’autres programmes puissent les utiliser sans connaître les détails internes de l’application.


\begin{center}
\begin{longtable}{|p{3.5cm}|p{5cm}|p{2.5cm}|p{4cm}|}
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Technologies} & \textbf{Critère de Validation} \\ \hline
\endfirsthead
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Technologies} & \textbf{Critère de Validation} \\ \hline
\endhead
Sélection et l'initialisation d'environnements. & L'API doit permettre la sélection et l'initialisation de différents environnements. & PettingZoo, MAgent2 & Tests unitaires : Initialiser un environnement de jeu OvercookedAI \\ \hline
Configuration d'algorithmes MARL/RL & L'API doit permettre de configurer les algorithmes RL/MARL (ex : PPO, MAPPO,etc...). & PyTorch, EPyMARL, Mava, JaxMARL, MARLLib & Tests unitaires : sélectionner un algorithme PPO en passant par les différents paramétrages \\ \hline
Gestion intégrée des phases & L'API doit permettre la pré-configuration de l'environnement ainsi que l'entraînement et les tests. & PettingZoo, MAgent2 & Tests unitaires : pré-configuration de l'environnement OvercookedAI, pré-configuration de l'entraînement avec l'algorithme PPO et des tests \\ \hline
\end{longtable}
\end{center}

## *Optimisation automatique des hyper-paramètres*

Les hyper-paramètres désignent les paramètres d’un modèle d’apprentissage  permettant d'optimiser la qualité et la rapidité de l’apprentissage. Sélectionner et régler les bonnes valeurs d'un hyper-paramètre sert à adapter l’apprentissage du  modèle aux contraintes et exigences du problème à résoudre.


\begin{center}
\begin{longtable}{|p{3.5cm}|p{5cm}|p{3cm}|p{4cm}|}
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Technologies} & \textbf{Critère de Validation} \\ \hline
\endfirsthead
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Technologies} & \textbf{Critère de Validation} \\ \hline
\endhead
Optimisation de l'efficacité d'apprentissage & Utilisation d'algorithmes pour l'optimisation automatique des hyperparamètres et arrêt des essais non prometteurs (élagage). & Librairie Optuna & Tests unitaires : exécuter l'optimisation des hyper-paramètres sur l'algorithme de PPO précédemment configuré. \\ \hline
Parallélisation des essais & Possibilité d'exécuter les essais de multiples agents en parallèle sans modification du code source. & Librairie Optuna & Tests unitaires : Configurer plusieurs essais d'entraînement et exécuter plusieurs optimisations simultanées des algorithmes lancés. \\ \hline
Visualisation (Optionnel) & Outils graphiques pour observer l'évolution et les résultats des optimisations effectuées. & Librairie Optuna & Tests unitaires : Afficher un graphe représentant deux hyper-paramètres sélectionnés montrant l'évolution des résultats. \\ \hline
\end{longtable}
\end{center}

## *Mise en place d’un guide pour les utilisateurs*

Un tutoriel/guide peut grandement aider un nouvel utilisateur qui souhaiterait utiliser une application qu’il ne connaît pas.

\begin{center}
\begin{longtable}{|p{4cm}|p{6.5cm}|p{4cm}|}
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Critère de Validation} \\ \hline
\endfirsthead
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Critère de Validation} \\ \hline
\endhead
Tutoriel écrit pour l'utilisateur (Wiki) & Apporter un guide complet pour un utilisateur ayant des connaissances techniques limitées. & Tests utilisateurs / Feedback: faire remplir une grille de satisfaction (note de 1 à 10 + commentaire optionnel) couvrant chaque étapes du wiki. \\ \hline
Complément vidéo & (Optionnel) Vidéo de démonstration pour illustrer la prise en main de l'outil. & Tests utilisateurs / Feedback \\ \hline
\end{longtable}
\end{center}

## *Mise en place d’une interface CLI / GUI*

Optionnellement une interface CLI voire GUI facilitant l’utilisation de l’application. Le but d’une interface CLI est d'exécuter l’application via des lignes de commandes (texte), un GUI permet de faire ce qu’un CLI fait mais grâce à une interface visuelle, rendant l’application plus accessible.


\begin{center}
\begin{longtable}{|p{3cm}|p{5cm}|p{2.5cm}|p{4cm}|}
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Technologies} & \textbf{Critère de Validation} \\ \hline
\endfirsthead
\hline
\textbf{Intitulé de la tâche} & \textbf{Description} & \textbf{Technologies} & \textbf{Critère de Validation} \\ \hline
\endhead
Commandes documentées (CLI) & Commandes de base nécessaires au lancement du programme, accompagnées de leur documentation. & argparse (librairie python) & Tests unitaires: tester que les commandes réalisent les actions souhaités \\ \hline
Gestion de l'environnement & Commandes spécifiques pour créer et sélectionner l'environnement (OvercookedAI, algorithme PPO) pour le système multi-agent. & argparse (librairie python) & Tests unitaires: tester que les commandes créent et séléctionnent l'environnement \\ \hline
Sélection des algorithmes & Commandes permettant de choisir un ou plusieurs algorithmes à intégrer aux agents. & argparse (librairie python) & Tests unitaires: tester que les bons algorithmes sont bien séléctionné avec les commandes \\ \hline
Paramétrage et optimisation & Commandes pour sélectionner, paramétrer et optimiser les hyperparamètres. & argparse (librairie python) & Tests unitaires: : Validation des sorties de commandes (paramétrage et optimisation réussit). \\ \hline
Commande de test et rendu du résultat & Commandes pour lancer les suites de tests et visualiser les rapports d'exécution. & argparse (librairie python) & Tests unitaires : tester la visualisation des rapports, qu'ils soient bien générés \\ \hline
Interface graphique Web & Interface visuelle permettant d'exécuter les commandes de manière intuitive. & Vue.js / React / Angular & Tests unitaires / fonctionnels: tests permettant de s'assurer que les interractions avec l'interface exécutent les bonnes actions\\ \hline
\end{longtable}
\end{center}

# Bibliographie
