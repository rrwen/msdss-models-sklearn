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

            # (get_sklearn_models_create_convert) Create converted model
            model_name = f'sklearn.{name}.{model.__name__}'
            can_update = False if name in ['preprocessing', 'random_projection'] else True
            converted_model = type(model.__name__, (Model,), {
                '__init__': create_init_method(can_update=can_update),
                'input': create_input_method(model),
                'output': create_output_method(),
                'update': create_update_method()
            })

            # (get_sklearn_models_create_del) Delete unneeded methods
            if not can_update:
                delattr(converted_model, 'update')

            # (get_sklearn_models_create_module) Add to module namespace
            setattr(sys.modules[base_module], model.__name__, converted_model)

            # (get_sklearn_models_create_docs) Add docs for converted model
            converted_model.__doc__ = f"""
            {model.__name__} from `scikit-learn <https://scikit-learn.org>`_.

            See `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}>`_.

            Parameters
            ----------
            settings : dict
                Optional initial settings for this model, consisting of the following keys:

                * ``x`` (list(str)): default independent variable names for ``input``, ``update``, and ``output``
                * ``y`` (str): default dependent predictor variable name for ``input``, ``update``, and ``output``
            kwargs : any
                Additional keyword arguments passed to `msdss_models_api.models.Model <https://rrwen.github.io/msdss-models-api/reference/models.html#model>`_.
            """

            # (get_sklearn_models_create_docs_input) Add docs for input method
            converted_model.input.__doc__ = f"""
            Initialize model with input data.

            See `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}>`_.
            
            Parameters
            ----------
            data : list(dict)
                Data to use for initializing the model. Should be a list of JSON-like objects (one for each row), where each key is a column name.
            x : list(str)
                Independent variable names to fit the model from parameter ``data``.
            y : str
                Dependent predictor variable name to fit the model from parameter ``data``.
            _fit : dict
                Additional keyword arguments passed to `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}#{model_name}.fit>`_
            kwargs : any
                Additional keyword arguments passed to `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}>`_.
            """

            # (get_sklearn_models_create_docs_output) Add docs for output method
            output_method = 'predict' if 'predict' in dir(model) else 'transform'
            converted_model.output.__doc__ = f"""
            Get model output.

            See `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}.{output_method}>`_.
            
            Parameters
            ----------
            data : list(dict)
                Data to get output from the model from. Should be a list of JSON-like objects (one for each row), where each key is a column name.
            x : list(str)
                Column names for independent variables from parameter ``data``. This can also be useful to reorder columns.
            y : str
                Column name for dependent predictor variable for output.
            kwargs : any
                Additional keyword arguments passed to `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}.{output_method}>`_.
            """

            # (get_sklearn_models_create_docs_output) Add docs for output method
            converted_model.update.__doc__ = f"""
            Update model with new data.

            See `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}#{model_name}.fit>`_.
            
            Parameters
            ----------
            data : list(dict)
                Data to update the model with. Should be a list of JSON-like objects (one for each row), where each key is a column name.
            x : list(str)
                Independent variable names to fit the model from parameter ``data``.
            y : str
                Dependent predictor variable name to fit the model from parameter ``data``.
            kwargs : any
                Additional keyword arguments passed to `{model_name} <https://scikit-learn.org/stable/modules/generated/{model_name}#{model_name}.fit>`_.
            """

            # (get_sklearn_models_create_out) Add converted model to dict
            out[model_name] = converted_model
    return out

models = get_sklearn_models(__name__)