import pandas

def create_input_method(model):
    def input(self, data, x=None, y=None, _fit={}, *args, **kwargs):
        data = pandas.DataFrame(data)
        data_x = data[x] if x else data
        data_y = data[y] if y else None
        self.instance = model(*args, **kwargs).fit(data_x, data_y, **_fit)
    return input

def create_output_method():
    def output(self, data, *args, **kwargs):
        data = pandas.DataFrame(data)
        model = self.instance
        if 'predict' in dir(model):
            out = pandas.DataFrame(model.predict(data, *args, **kwargs))
        else:
            out = pandas.DataFrame(model.transform(data, *args, **kwargs))
        return out
    return output

def create_update_method():
    def update(self, data, x, y, *args, **kwargs):
        data = pandas.DataFrame(data)
        data_x = data[x]
        data_y = data[y]
        self.instance.fit(data_x, data_y, *args, **kwargs)
    return update
    