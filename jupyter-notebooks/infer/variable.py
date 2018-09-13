class Variable(object):
    def __init__(self, name, prior=None, size=1):
        self.name = name
        self.size = size
        self.prior = prior
        self.value = None
        self.sample_size = max(1, int(size*0.01))
