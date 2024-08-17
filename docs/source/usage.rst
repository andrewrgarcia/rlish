Usage
=====

Saving Data
-----------

To save data, use the `save` function. You can choose between `pickle` and `joblib` formats:

.. code-block:: python

    import rlish
    import numpy as np

    dictionary = {'a': 1, 'b': 2, 'c': 3}
    tensor = np.random.randint(0, 10, (200, 200, 200))

    # Save dictionary using pickle
    rlish.save(dictionary, 'my_dictio')

    # Save data using joblib
    rlish.save(tensor, 'huge_tensor', format='joblib')

Loading Data
------------

To load data, use the `load` function:

.. code-block:: python

    # Load data saved with pickle
    loaded_data_pickle = rlish.load('my_dictio')

    # Load data saved with joblib
    loaded_data_joblib = rlish.load('huge_tensor')

    # Load your data with the format printed out (if you forgot)
    loaded_data_joblib = rlish.load('huge_tensor', what_is=True)
