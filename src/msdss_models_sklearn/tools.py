import inspect
import pandas
import sklearn
import sys

from msdss_models_api.models import Model
from sklearn import *

from .defaults import *

def create_input_method(model):
    def input(self, data, x=None, y=None, _fit={}, *args, **kwargs):
        
        # (create_input_method_vars) Set default vars
        x = x if x else self.settings['x'] if 'x' in self.settings else x
        y = y if y else self.settings['y'] if 'y' in self.settings else y

        # (create_input_method_data) Format data for model instance input
        data = pandas.DataFrame(data)
        data_x = data[x] if x else data
        data_y = data[y] if y else None

        # (create_input_method_set) Train model instance
        self.instance = model(*args, **kwargs).fit(data_x, data_y, **_fit)
    return input

def create_output_method():
    def output(self, data, x=None, y=None, *args, **kwargs):

        # (create_output_method_vars) Set default vars
        x = x if x else self.settings['x'] if 'x' in self.settings else x
        y = y if y else self.settings['y'] if 'y' in self.settings else y
        y = [y] if y and not isinstance(y, list) else y

        # (create_output_method_data) Format data for model instance output
        data = pandas.DataFrame(data)
        data_x = data[x] if x else data

        # (create_output_method_output) Get output from trained model instance
        model = self.instance
        if 'predict' in dir(model):
            out = pandas.DataFrame(model.predict(data_x, *args, **kwargs))
        else:
            out = pandas.DataFrame(model.transform(data_x, *args, **kwargs))
        
        # (create_output_method_return) Set column names if avail and return output
        if y:
            out.columns = y
        return out
    return output

def create_update_method():
    def update(self, data, x=None, y=None, *args, **kwargs):

        # (create_update_method_vars) Set default vars
        x = x if x else self.settings['x'] if 'x' in self.settings else x
        y = y if y else self.settings['y'] if 'y' in self.settings else y

        # (create_update_method_data) Format data for update
        data = pandas.DataFrame(data)
        data_x = data[x] if x else data
        data_y = data[y] if y else None

        # (create_update_method_update) Update model instance
        self.instance.fit(data_x, data_y, *args, **kwargs)
    return update
    
def get_sklearn_models(modules=DEFAULT_MODULES):
    out = {}
    for name in modules:

        # (get_sklearn_models_module) Get the sklearn module
        module = getattr(sklearn, name)

        # (get_sklearn_models_filter1) Filter for first letter caps and no underscore in module namespace
        module_models = [getattr(module, m) for m in dir(module) if m[0] == m[0].upper() and m[0] != '_']

        # (get_sklearn_models_filter2) Filter for classes that are inside the module only
        module_models = [m for m in module_models if inspect.isclass(m) and m.__module__.startswith(f'sklearn.{name}')]

        # (get_sklearn_models_create) Create msdss model and add to dict
        for model in module_models:
            model_name = f'sklearn.{name}.{model.__name__}'
            out[model_name] = type(model.__name__, (Model,), {
                'input': create_input_method(model),
                'output': create_output_method(),
                'update': create_update_method()
            })
            setattr(sys.modules[__name__], model.__name__, out[model_name])
    return out