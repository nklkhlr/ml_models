class BaseModel:
    """
    Prototype class for all models in order to be compatible
    with training implementation
    """
    def __init__(self, *args, **kwargs):
        pass

    @property
    def parameters(self):
        raise NotImplemented

    def train(self, *args, **kwargs):
        raise NotImplemented

    def update_parameters(self, *args, **kwargs):
        raise NotImplemented
