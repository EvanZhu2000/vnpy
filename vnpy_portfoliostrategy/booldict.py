# An efficient implementation tracking whether all positions have been adjusted to desire

class BoolDict:
    def __init__(self, false_keys_list: list):
        self.data = {}
        self.true_count = 0
        for k in false_keys_list:
            self.set(k, False)

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
    
    def get_false_keys(self):
        return [key for key, value in self.data.items() if not value]