# An efficient implementation tracking whether all positions have been adjusted to desire

class BoolDict:
    def __init__(self):
        self.data = {}
        self.true_count = 0

    def set(self, key, value):
        # If the value is being changed from False to True
        if key not in self.data and value:
            self.true_count += 1
        elif key in self.data:
            if not self.data[key] and value:
                self.true_count += 1
            elif self.data[key] and not value:
                self.true_count -= 1
        self.data[key] = value

    def all_true(self):
        return self.true_count == len(self.data)