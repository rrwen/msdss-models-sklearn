import inspect
import sklearn

from msdss_models_api.models import Model
from sklearn import *

from .defaults import *
from .tools import *

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
    return out