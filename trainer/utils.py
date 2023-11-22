class Accuracy():
    def __init__(self):
        self.avg = 0
        self.reset()
    @property
    def accuracy(self):
        return self.avg
    def reset(self):
        self.count = 0

    def __call__(self, value):
        value = self.count * self.avg + value
        self.count += 1
        self.avg = value / self.count
        return self.avg