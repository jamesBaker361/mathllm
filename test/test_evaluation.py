import unittest

class EvalTest(unittest.TestCase):
    def test_evaluate_wandb(self):
        '''
        when given the test dataset
        for each model type (normal, peft, rl) loads model
        and uploads metrics to wandb
        '''
        return