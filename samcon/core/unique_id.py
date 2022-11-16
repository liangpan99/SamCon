class UniqueId():

    def __init__(self):
        self._id = 0
    
    def get(self, num=1):
        ids = range(self._id, self._id + num)
        self._id += num
        return iter(ids)

    def reset(self):
        self._id = 0
