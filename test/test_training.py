import unittest

class TrainingTest(unittest.TestCase):
    def test_train_wandb(self):
        '''
        when given the train dataset
        and n epochs
        and model type (normal, peft, rl) trains model
        and uploads metrics to wandb
        '''
        return