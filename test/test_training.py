import unittest
import os
import sys
sys.path.append(os.getcwd())
from training import *

class TrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["WANDB_PROJECT"]="math-llm-testing"

    def test_train_wandb_dummy(self):
        '''
        when given the train dataset parameters
        and n epochs
        and model type =ft trains model
        and uploads metrics to wandb
        '''
        for training_type_list in [[FT]]:
            with self.subTest(training_type_list=training_type_list):
                for task_list in [["dumb"]]:
                    with self.subTest(task_list=task_list):
                        for number_type_list in [[WHOLE], [WHOLE, DECIMAL], [DECIMAL]]:
                            with self.subTest(number_type_list=number_type_list):
                                training_loop(1, training_type_list, task_list, number_type_list)

if __name__=='__main__':
    test_case=TrainingTest()
    test_case.setUp()
    test_case.test_train_wandb_dummy()