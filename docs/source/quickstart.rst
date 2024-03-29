Quick Start
===========

Obtaining Converted Models
--------------------------

After installing the package, get the converted models as a dict from the models variable:

.. jupyter-execute::

    from msdss_models_sklearn import models as sklearn_models

    sklearn_models['sklearn.linear_model.LinearRegression']

For more details, see :func:`msdss_models_sklearn.core.get_sklearn_models`.

.. note::

   The extracted models can also be imported directly from the package if you do not require all the models in your API:

   .. jupyter-execute::
        
        from msdss_models_sklearn import LinearRegression, DecisionTreeClassifier

        selected_models = [LinearRegression, DecisionTreeClassifier]
        selected_models

Adding Converted Models to Models API
-------------------------------------

To add the converted scikit-learn models to the msdss-models-api, set the models argument to the imported models:

.. code-block:: python
    
    from msdss_models_api import ModelsAPI
    from msdss_models_sklearn import models as sklearn_models

    # Create app using env vars
    app = ModelsAPI(models=sklearn_models)

    # Get the redis background worker to run using celery
    worker = app.get_worker()

See `msdss-models-api Quick Start <https://rrwen.github.io/msdss-models-api/quickstart.html>`_ for more details on running the API.

.. note::

    Ensure that `msdss-models-api <https://rrwen.github.io/msdss-models-api>`_ has been installed and setup properly for the API to work.
    
    See `msdss-models-api Install <https://rrwen.github.io/msdss-models-api/install.html>`_.
