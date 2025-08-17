'''
Contains a variety of tests to validate the efficacy of the classifier
models in the current assignment.
'''
import pandas as pd
import math
import itertools
import unittest
import pytest
import copy
from toxicity_filter import *
from salary_predictor import *
from sklearn.model_selection import train_test_split # type: ignore

# Test Constants
# -------------------------------------------------------------------------------------------------
# Set VERBOSE to True to see informative outputs from steps of tests, 
# or False just to check correctness
VERBOSE = True

# Toxicity Filter Constants
# -------------------------------------------------------------------------------------------------
TOXICITY_DATA = pd.read_csv("../dat/wiki_talk.csv")
TOX_TEXT_TRAIN, TOX_TEXT_TEST, TOX_LABELS_TRAIN, TOX_LABELS_TEST = train_test_split(TOXICITY_DATA["comment"], TOXICITY_DATA["toxic"], test_size=0.3, random_state=1337)
TOX_TEXT_TRAIN_SIZE, TOX_LABEL_TRAIN_SIZE = len(TOX_TEXT_TRAIN), len(TOX_LABELS_TRAIN)

# Salary Predictor Constants
# -------------------------------------------------------------------------------------------------
SALARY_DATA = pd.read_csv('../dat/salary.csv')
SAL_X_TRAIN, SAL_X_TEST, SAL_Y_TRAIN, SAL_Y_TEST = train_test_split(SALARY_DATA.drop(axis=1, columns=['class']), SALARY_DATA['class'], test_size=0.3, random_state=42)
SAL_X_TRAIN_SIZE, SAL_Y_TRAIN_SIZE = len(SAL_X_TRAIN), len(SAL_Y_TRAIN)


class ClassiferTests(unittest.TestCase):
    
    # Test Helpers
    # -------------------------------------------------------------------------------------------------
    def verify_data(self, X: pd.DataFrame, Y: pd.DataFrame, x_len: int, y_len: int) -> None:
        DAT_ERR = "[X] Make sure your constructor isn't tampering with the data frame itself!"
        self.assertEqual(x_len, len(X), DAT_ERR)
        self.assertEqual(y_len, len(Y), DAT_ERR)
        
        
    # Toxicity Filter Tests
    # -------------------------------------------------------------------------------------------------
    def test_toxicity_filter_vectorizer(self) -> None:
        self.verify_data(TOX_TEXT_TRAIN, TOX_LABELS_TRAIN, TOX_TEXT_TRAIN_SIZE, TOX_LABEL_TRAIN_SIZE)
        talk_xic = ToxicityFilter(TOX_TEXT_TRAIN, TOX_LABELS_TRAIN)
        self.assertTrue(talk_xic.vectorizer is not None, "[X] Make sure your vectorizer is initialized in the constructor")
        self.assertTrue(hasattr(talk_xic.vectorizer, "vocabulary_"), "[X] Make sure your vectorizer is .fit on the text_train parameter in your constructor")
        vocab = talk_xic.vectorizer.vocabulary_
        if VERBOSE:
            print("=======================================")
            print("[Test] test_toxicity_filter_vectorizer")
            print("  > Vectorizer Vocabulary Length: " + str(len(vocab)))
            print("  > Vectorizer Vocabulary Sample (Randomish 30): " + str(list(vocab.items())[0:30]))
        
        # Should be closer to 150k but leaving some lenience for whether or not you added more pruning of words
        self.assertLess(100000, len(vocab))
    
    def test_toxicity_filter_classification(self) -> None:
        self.verify_data(TOX_TEXT_TRAIN, TOX_LABELS_TRAIN, TOX_TEXT_TRAIN_SIZE, TOX_LABEL_TRAIN_SIZE)
        talk_xic = ToxicityFilter(TOX_TEXT_TRAIN, TOX_LABELS_TRAIN)
        
        test_comments = [
            "yo this article is straight booty poop are the mods sleeping",
            "I agree with you completely, thank you for this cordial exchange"
        ]
        classes_assigned = []
        try:
            classes_assigned = talk_xic.classify(test_comments)
        except ValueError:
            raise ValueError("[X] Did you use the fit_transform method of your vectorizer in classify rather than just transform? (Should only be fit once in constructor)")
        
        labels_assigned = ["TOXIC" if classes_assigned[i] else "Non-toxic" for i in range(len(classes_assigned))]
        solution = ["TOXIC", "Non-toxic"]
        
        if VERBOSE:
            print("=======================================")
            print("[Test] test_toxicity_filter_classification")
        for index, comment in enumerate(test_comments):
            if VERBOSE:
                print("  [" + str(index) + "] " + comment + " -> " + labels_assigned[index])
            self.assertEqual(solution[index], labels_assigned[index])
     
    def test_toxicity_filter_performance(self) -> None:
        self.verify_data(TOX_TEXT_TRAIN, TOX_LABELS_TRAIN, TOX_TEXT_TRAIN_SIZE, TOX_LABEL_TRAIN_SIZE)
        talk_xic = ToxicityFilter(TOX_TEXT_TRAIN, TOX_LABELS_TRAIN)
        (report_str, report_dict) = talk_xic.test_model(TOX_TEXT_TEST, TOX_LABELS_TEST)
        if VERBOSE:
            print("=======================================")
            print("[Test] test_toxicity_filter_performance")
            print(report_str)
        
        report_acc: float = report_dict["accuracy"]
        self.assertLess(0.92, report_acc)
    
    
    # Salary Predictor Tests
    # -------------------------------------------------------------------------------------------------
    def test_salary_preprocessing(self) -> None:
        self.verify_data(SAL_X_TRAIN, SAL_Y_TRAIN, SAL_X_TRAIN_SIZE, SAL_Y_TRAIN_SIZE)
        pre_preprocessing_x_train = copy.deepcopy(SAL_X_TRAIN)
        pre_preprocessing_x_test = copy.deepcopy(SAL_X_TEST)
        sal_pred = SalaryPredictor(SAL_X_TRAIN, SAL_Y_TRAIN)
        
        # Didn't mess with the original dfs did you?
        MOD_ERR = "[X] You're not allowed to modify the original dataframe -- make a copy if you need to change anything."
        self.assertTrue(pre_preprocessing_x_train.equals(SAL_X_TRAIN), MOD_ERR)
        features = sal_pred.preprocess(SAL_X_TEST, False)
        self.assertTrue(pre_preprocessing_x_test.equals(SAL_X_TEST), MOD_ERR)
        
        # Here you have some flexibility on what actually got returned, but want to make sure at least
        # the number of rows of the ndarray are correct
        self.assertEqual(len(pre_preprocessing_x_test), features.shape[0])
    
    def salary_performance_setup(self) -> float:
        self.verify_data(SAL_X_TRAIN, SAL_Y_TRAIN, SAL_X_TRAIN_SIZE, SAL_Y_TRAIN_SIZE)
        sal_pred = SalaryPredictor(SAL_X_TRAIN, SAL_Y_TRAIN)
        (report_str, report_dict) = sal_pred.test_model(SAL_X_TEST, SAL_Y_TEST)
        if VERBOSE:
            print("=======================================")
            print("[Test] test_salary_performance")
            print(report_str)
        
        report_acc: float = report_dict["accuracy"]
        return report_acc
    
    def test_salary_performance_easy(self) -> None:
        self.assertLess(0.80, self.salary_performance_setup())
        
    def test_salary_performance_med(self) -> None:
        self.assertLess(0.82, self.salary_performance_setup())
        
    def test_salary_performance_hard(self) -> None:
        self.assertLess(0.84, self.salary_performance_setup())
    
    # [!] For grading, the same 3 tests will be repeated, but for an unseen test set!
        
if __name__ == '__main__':
    unittest.main()
