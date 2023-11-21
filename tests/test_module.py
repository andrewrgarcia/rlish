import unittest
import os
from rlish.data_io import save, load
import numpy as np

class TestRlishModule(unittest.TestCase):
    def setUp(self):
        """Setup a temporary dataset for testing."""
        self.test_data = {'a': 1, 'b': 2, 'c': 3}
        self.test_joblib = np.random.randint(0,10,(200,200,200))

        self.filename_pickle = 'test_data.pkl'
        self.filename_joblib = 'test_data.joblib'

    def test_save_and_load_pickle(self):
        """Test the save and load functions with pickle."""
        # Save data using pickle
        save(self.test_data, self.filename_pickle, format='pickle')

        # Load data
        loaded_data = load(self.filename_pickle)
        print(loaded_data)

        # Check if the data matches
        self.assertEqual(self.test_data, loaded_data)

        # Cleanup
        os.remove(self.filename_pickle)
        os.remove(self.filename_pickle + '.meta')

    def test_save_and_load_joblib(self):
        """Test the save and load functions with joblib."""
        # Save data using joblib
        save(self.test_joblib, self.filename_joblib, format='joblib')

        # Load data
        loaded_data = load(self.filename_joblib)
        print(loaded_data)

        # Check if the data matches
        np.allclose(self.test_joblib, loaded_data)
        # Cleanup
        os.remove(self.filename_joblib)
        os.remove(self.filename_joblib + '.meta')

    # Add more tests here if needed

if __name__ == '__main__':
    unittest.main()
