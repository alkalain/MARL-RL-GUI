from abc import ABC, abstractmethod

def patch_marllib():
    """
    Applique un monkey-patch sur MARLlib pour autoriser l'injection de paramètres d'environnement dynamiques.

    Par défaut, MARLlib utilise une fonction interne `dict_update` dotée d'une vérification stricte.
    Celle-ci rejette systématiquement tout argument (ex: `num_good`, `num_adversaries`) qui n'est pas
    explicitement déclaré dans ses propres fichiers de configuration `.yaml`.

    Ce patch remplace cette fonction à l'exécution par une version tolérante, permettant à
    l'utilisateur de modifier la configuration de l'environnement à la volée (via `**kwargs`)
    sans lever de `ValueError`.

    **Mécanisme d'action :**
    Pour contourner les problèmes de liaison d'import (Import Binding), la fonction écrase
    la référence stricte de `dict_update` dans tous les espaces de noms (namespaces) critiques :

    1. Le module source d'origine (`marllib.marl.common`).
    2. L'espace de noms d'initialisation (`marllib.marl`), qui est la référence copiée
       et utilisée par `make_env()` à la ligne 102 de `__init__.py`.
    3. Le sous-module `marllib.marl.marl` par mesure de sécurité.

    Notes:
        Cette fonction doit impérativement être appelée **avant** toute instanciation
        d'environnement (notamment juste avant `marl.make_env()`) pour s'assurer que le
        patch écrase bien les références en mémoire au bon moment.

    Raises:
        ImportError: L'exception est interceptée en interne de manière silencieuse si
            le module `marllib` n'est pas installé ou introuvable. Un avertissement
            est simplement affiché dans la console et l'exécution se poursuit.
    """
    def tolerant_dict_update(d, u, *args, **kwargs):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = tolerant_dict_update(d[k], v)
            else:
                d[k] = v
        return d

    try:
        import marllib
        import marllib.marl.common

        marllib.marl.common.dict_update = tolerant_dict_update

        marllib.marl.dict_update = tolerant_dict_update

        try:
            import marllib.marl.marl
            marllib.marl.marl.dict_update = tolerant_dict_update
        except ImportError:
            pass

        print("[MARIO] Monkey-patch MARLlib appliqué (Namespaces écrasés avec succès).")
    except ImportError as e:
        print(f"[MARIO] [Attention] Impossible d'appliquer le patch MARLlib : {e}")

class JointPolicy(ABC):
    """
    Interface abstraite représentant la politique jointe des agents.
    
    Cette classe sert de base pour encapsuler les modèles entraînés. Elle définit 
    comment une entité de décision doit transformer les perceptions globales 
    en actions individuelles pour chaque agent.
    """
    def __init__(self, policy_type: str):
        """
        Initialise le type de politique.

        Args:
            policy_type (str): Étiquette identifiant la nature de la politique 
                (ex: "MARLlib_Policy", "Scripted_Policy").
        """
        self.type = policy_type

    @abstractmethod
    def predict(self, observations):
        """
        Détermine les actions à exécuter en fonction des observations actuelles.

        Args:
            observations: État perçu de l'environnement (format dépendant du moteur utilisé).

        Returns:
            dict: Un dictionnaire associant l'identifiant de chaque agent à son action 
                correspondante (ex: {'agent_0': 1, 'agent_1': 0}).
        """
        pass

class Algo(ABC):
    """
    Interface de base pour la configuration et la gestion des algorithmes.
    
    Cette classe définit le cadre nécessaire pour implémenter des algorithmes 
    d'apprentissage (ex: PPO, MAPPO) de manière indépendante des autres modules.
    """
    def __init__(self, algo_type: str):
        """
        Initialise les propriétés fondamentales de l'algorithme.

        Args:
            algo_type (str): Nom explicite de l'algorithme utilisé.
        """
        self.type = algo_type

class ArchitectureSupport(ABC):
    """
    Interface dédiée à la structure des réseaux de neurones.
    
    Elle permet de définir la configuration structurelle (couches, types de neurones) 
    qui sera injectée dans l'algorithme pour construire le modèle d'apprentissage.
    """
    def __init__(self, arch_type: str):
        """
        Initialise le type d'architecture.

        Args:
            arch_type (str): Identifiant de l'architecture (ex: "MLP", "CNN", "RNN").
        """
        self.type = arch_type

    @abstractmethod
    def to_marllib_config(self) -> dict:
        """
        Convertit les paramètres de l'objet en un format compatible avec MARLlib.

        Returns:
            dict: Configuration prête à être transmises aux fonctions de 
                construction de modèle de MARLlib.
        """
        pass