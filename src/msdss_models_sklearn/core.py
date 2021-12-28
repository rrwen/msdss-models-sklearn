import inspect
import sklearn
import sys

from msdss_models_api.models import Model
from sklearn import *

from .defaults import *
from .tools import *

def get_sklearn_models(base_module=__name__, modules=DEFAULT_MODULES):
    """
    Extract scikit-learn models and convert them into :class:`msdss_models_api:msdss_models_api.models.Model` classes with appropriate methods for input, output, and update.

    To extract scikit-learn models, the general process is:

    1. Get the models module (modules are manually defined from inspecting the docs)
    2. Filter out all names in the module where the first letter is not capitalized or contains an underscore
    3. Filter out all names in the module that are not classes or are not part of the module
    4. Dynamically create a class for the resulting names, extending from :class:`msdss_models_api:msdss_models_api.models.Model`
    5. Add methods:
        
        * ``input`` using :func:`msdss_models_sklearn.tools.create_input_method`
        * ``output`` using :func:`msdss_models_sklearn.tools.create_output_method`
        * ``update`` using :func:`msdss_models_sklearn.tools.create_update_method`
    
    Parameters
    ----------
    base_module : str
        Name of the base module to attach model namespaces to.
    modules : list(str)
        List of scikit-learn module names to extract models from.

    Author
    ------
    Richard Wen <rrwen.dev@gmail.com>
    
    Example
    -------
    .. jupyter-execute::

        from msdss_models_sklearn import models as sklearn_models
        from pprint import pprint

        # Print all converted models
        pprint(sklearn_models)
    """
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
            setattr(sys.modules[base_module], model.__name__, out[model_name])
    return out

models = get_sklearn_models(__name__)