
# define a class used to set global constants
class GlobalSetting(object):
    """
        global setting
    """
    def __init__(self) -> None:
        """
            This function is empty
        """
        pass

    @staticmethod
    def set() -> None:
        """
        global settings
        :return:
        """
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        seed = 777
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        return None
