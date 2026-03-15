class TrainingParamsStop:
    """Définit les critères pour arrêter un entraînement prématurément (élagage d'Optuna)."""
    def __init__(self, stop_type: str):
        self.type = stop_type