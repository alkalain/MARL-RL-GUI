class Stats:
    """Stocke les métriques d'évaluation (taux de victoire, récompenses moyennes, etc.)."""
    def __init__(self, stats_type: str):
        self.type = stats_type