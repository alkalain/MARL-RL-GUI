class TrainingParamsStop:
    """
    Gestionnaire des critères d'arrêt pour les sessions d'apprentissage.
    
    Cette classe définit les conditions permettant d'interrompre un entraînement, 
    soit par l'atteinte d'un objectif, soit via un mécanisme d'élagage (pruning) 
    pour stopper les essais dont les performances sont jugées insuffisantes.
    """
    def __init__(self, stop_type: str):
        """
        Initialise le critère d'arrêt.

        Args:
            stop_type (str): Nature du critère (ex: "Iteration_Limit", "Early_Stopping").
        """
        self.type = stop_type