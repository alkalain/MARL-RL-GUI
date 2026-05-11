class Stats:
    """
    Conteneur dédié au stockage et à la gestion des métriques d'évaluation.
    
    Cette classe permet de regrouper les indicateurs de performance collectés 
    durant les phases de test ou d'entraînement (ex: taux de victoire, 
    récompenses cumulées, longueur des épisodes).
    """
    def __init__(self, stats_type: str):
        """
        Initialise un collecteur de statistiques.

        Args:
            stats_type (str): Catégorie des statistiques collectées 
                (ex: "Evaluation", "Training").
        """
        self.type = stats_type