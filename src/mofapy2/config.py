#
# Global configuration
#


class MOFAConfig:
    """
    MOFA config manager
    """

    def __init__(self, use_float32: bool = False):
        self.use_float32 = use_float32

    @property
    def use_float32(self):
        return self.float32

    @use_float32.setter
    def use_float32(self, float32):
        if not isinstance(float32, bool):
            msg = "True or False has to be provided."
            raise TypeError(msg)
        self.float32 = float32


config = MOFAConfig()
