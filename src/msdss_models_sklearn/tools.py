import pandas

from msdss_models_api.models import Model

def create_init_method(can_input=True, can_output=True, can_update=True):
    """
    Create model init method for scikit-learn models to be compatible with :class:`msdss_models_api:msdss_models_api.models.Model`.

    See :class:`msdss_models_api:msdss_models_api.models.Model`.

    Parameters
    ----------
    can_input : bool
        Whether the method ``.input`` is defined and available. This is useful for controlling route requests in an API.
    can_output : bool
        Whether the method ``.output`` is defined and available. This is useful for controlling route requests in an API.
    can_update : bool
        Whether the method ``.update`` is defined and available. This is useful for controlling route requests in an API.

    Author
    ------
    Richard Wen <rrwen.dev@gmail.com>
    
    Example
    -------
    .. jupyter-execute::

        from msdss_models_sklearn.tools import *
        from sklearn.linear_model import LinearRegression

        input = create_input_method(LinearRegression)
    """
    def init(self, can_input=can_input, can_output=can_output, can_update=can_update, *args, **kwargs):
        Model.__init__(self, can_input=can_input, can_output=can_output, can_update=can_update, *args, **kwargs)
    return init

def create_input_method(model):
    """
    Create model input method for scikit-learn models to be compatible with :class:`msdss_models_api:msdss_models_api.models.Model`.

    See :meth:`msdss_models_api:msdss_models_api.models.Model.input`.

    Parameters
    ----------
    model : class
        Scikit-learn model class to create machine learning models.

    Author
    ------
    Richard Wen <rrwen.dev@gmail.com>
    
    Example
    -------
    .. jupyter-execute::

        from msdss_models_sklearn.tools import *
        from sklearn.linear_model import LinearRegression

        input = create_input_method(LinearRegression)
    """
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
    """
    Create model output method for scikit-learn models to be compatible with :class:`msdss_models_api:msdss_models_api.models.Model`.

    See :meth:`msdss_models_api:msdss_models_api.models.Model.output`.

    Author
    ------
    Richard Wen <rrwen.dev@gmail.com>
    
    Example
    -------
    .. jupyter-execute::

        from msdss_models_sklearn.tools import *

        output = create_output_method()
    """
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
    """
    Create model update method for scikit-learn models to be compatible with :class:`msdss_models_api:msdss_models_api.models.Model`.

    See :meth:`msdss_models_api:msdss_models_api.models.Model.update`.

    Author
    ------
    Richard Wen <rrwen.dev@gmail.com>
    
    Example
    -------
    .. jupyter-execute::

        from msdss_models_sklearn.tools import *

        update = create_output_method()
    """
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
