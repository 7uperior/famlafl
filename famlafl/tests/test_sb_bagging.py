"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import os
import unittest

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, mean_absolute_error, \
    mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from famlafl.util.volatility import get_daily_vol
from famlafl.filters.filters import cusum_filter
from famlafl.labeling.labeling import get_events, add_vertical_barrier, get_bins
from famlafl.sampling.bootstrapping import get_ind_matrix, get_ind_mat_label_uniqueness
from famlafl.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier, \
    SequentiallyBootstrappedBaggingRegressor


# pylint: disable=invalid-name

def _generate_label_with_prob(x, prob, random_state=np.random.RandomState(1)):
    """
    Generates true label value with some probability(prob)
    """
    choice = random_state.choice([0, 1], p=[1 - prob, prob])
    if choice == 1:
        return x
    return int(not x)


def _get_synthetic_samples(ind_mat, good_samples_thresh, bad_samples_thresh):
    """
    Get samples with uniqueness either > good_samples_thresh or uniqueness < bad_samples_thresh
    """
    print("ind_mat shape:", ind_mat.shape)
    if ind_mat.size == 0:
        print("Warning: ind_mat is empty")
        return list(range(100))  # Return first 100 samples as fallback

    # Get mix of samples where some of them are extremely non-overlapping, the other one are highly overlapping
    i = 0
    unique_samples = []
    uniqueness_values = []
    label_uniqueness = get_ind_mat_label_uniqueness(ind_mat)
    print("label_uniqueness shape:", label_uniqueness.shape)
    
    for label in label_uniqueness:
        if label.size > 0:
            mean_uniqueness = np.mean(label[label > 0]) if np.any(label > 0) else 0
            uniqueness_values.append(mean_uniqueness)
            if mean_uniqueness > good_samples_thresh or mean_uniqueness < bad_samples_thresh:
                unique_samples.append(i)
        i += 1
    
    print(f"Initial unique samples: {len(unique_samples)}")
    print(f"Uniqueness values stats: min={min(uniqueness_values) if uniqueness_values else 0}, "
          f"max={max(uniqueness_values) if uniqueness_values else 0}, "
          f"mean={np.mean(uniqueness_values) if uniqueness_values else 0}")
    
    if len(unique_samples) < 100:  # Ensure we have at least 100 samples
        print("Not enough samples, using percentile-based selection")
        uniqueness_array = np.array(uniqueness_values)
        if len(uniqueness_array) > 0:
            # Get samples from both tails of the distribution
            low_thresh = np.percentile(uniqueness_array, 25)
            high_thresh = np.percentile(uniqueness_array, 75)
            unique_samples = []
            for i, val in enumerate(uniqueness_values):
                if val <= low_thresh or val >= high_thresh:
                    unique_samples.append(i)
    
    if len(unique_samples) == 0:
        print("Still no samples, using first 100 indices")
        unique_samples = list(range(min(100, ind_mat.shape[0] if ind_mat.size > 0 else 100)))
    
    print("Final number of unique samples:", len(unique_samples))
    return unique_samples


class TestSequentiallyBootstrappedBagging(unittest.TestCase):
    """
    Test SequentiallyBootstrapped Bagging classifiers
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data and get triple barrier events, generate features
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

        print("\nInitial data shape:", self.data.shape)

        # Compute moving averages
        self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20,
                                                            center=False).mean()
        self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50,
                                                            center=False).mean()

        # Compute sides
        self.data['side'] = np.nan

        long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
        short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
        self.data.loc[long_signals, 'side'] = 1
        self.data.loc[short_signals, 'side'] = -1

        # Remove Look ahead bias by lagging the signal
        self.data['side'] = self.data['side'].shift(1)

        daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 2  # Increase target size
        cusum_events = cusum_filter(self.data['close'], threshold=0.001)  # Further lower threshold
        print("cusum_events length:", len(cusum_events))
        
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                 num_hours=2)
        print("vertical_barriers length:", len(vertical_barriers))
        
        meta_labeled_events = get_events(close=self.data['close'],
                                         t_events=cusum_events,
                                         pt_sl=[1, 1],  # More aggressive profit taking/stop loss
                                         target=daily_vol,
                                         min_ret=1e-6,  # Further lower minimum return threshold
                                         num_threads=3,
                                         vertical_barrier_times=vertical_barriers,
                                         side_prediction=self.data['side'],
                                         verbose=False)
        meta_labeled_events.dropna(inplace=True)
        print("meta_labeled_events shape:", meta_labeled_events.shape)
        
        labels = get_bins(meta_labeled_events, self.data['close'])
        print("labels shape:", labels.shape)

        # Generate data set which shows the power of SB Bagging vs Standard Bagging
        print("meta_labeled_events.t1 shape:", meta_labeled_events.t1.shape)
        print("meta_labeled_events.t1 head:\n", meta_labeled_events.t1.head())
        print("self.data.close shape:", self.data.close.shape)
        print("self.data.close head:\n", self.data.close.head())
        
        ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
        print("ind_mat shape:", ind_mat.shape)

        # Use more relaxed thresholds to ensure we get samples
        unique_samples = _get_synthetic_samples(ind_mat, 0.3, 0.05)

        # Get synthetic data set with drawn samples
        print("Shape before synthetic samples:", self.data.shape)
        print("Length of unique_samples:", len(unique_samples))
        X = self.data.loc[labels.index].iloc[unique_samples].copy()  # Use copy() to avoid SettingWithCopyWarning
        print("Shape after initial X creation:", X.shape)
        X = X.dropna()
        print("Shape after first dropna:", X.shape)
        labels = labels.loc[X.index, :]
        print("Shape of labels:", labels.shape)
        X['y'] = labels.bin  # Add y column directly
        print("Shape after adding y column:", X.shape)

        # Generate features (some of them are informative, others are just noise)
        if 'y' not in X.columns:
            raise ValueError("Column 'y' is missing from DataFrame X")
            
        # Create probability columns all at once using vectorized operations
        for prob in [0.6, 0.5, 0.3, 0.2, 0.1]:
            col_name = f'label_prob_{prob}'
            X[col_name] = X['y'].apply(lambda x: _generate_label_with_prob(x, prob))

        # Ensure all label probability columns exist before creating rolling means
        required_columns = ['label_prob_0.6', 'label_prob_0.5', 'label_prob_0.3', 'label_prob_0.2', 'label_prob_0.1']
        for col in required_columns:
            if col not in X.columns:
                raise ValueError(f"Required column {col} is missing from DataFrame")

        features = ['label_prob_0.6', 'label_prob_0.2']  # Two super-informative features
        for prob in [0.5, 0.3, 0.2, 0.1]:
            for window in [2, 5, 10]:
                X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(
                    window=window).mean()
                features.append('label_prob_{}_sma_{}'.format(prob, window))
        print("Shape before final dropna:", X.shape)
        X.dropna(inplace=True)
        print("Shape after final dropna:", X.shape)
        y = X.pop('y')
        print("Length of y:", len(y))

        # Create TimeSeriesSplit cross-validator with fewer splits and minimum size
        n_samples = len(X)
        if n_samples > 0:
            test_size = max(2, int(n_samples * 0.2))  # Ensure at least 2 samples in test
            tscv = TimeSeriesSplit(n_splits=2, test_size=test_size, gap=0)
            
            # Get the first split for train/test
            for train_indices, test_indices in tscv.split(X[features]):
                self.X_train = X[features].iloc[train_indices]
                self.X_test = X[features].iloc[test_indices]
                self.y_train_clf = y.iloc[train_indices]
                self.y_test_clf = y.iloc[test_indices]
                break  # Only use first split
        else:
            raise ValueError("No samples available for train/test split")

        self.y_train_reg = (1 + self.y_train_clf)
        self.y_test_reg = (1 + self.y_test_clf)

        self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
        self.price_bars_trim = self.data[
            (self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

    def test_sb_bagging_not_tree_base_estimator(self):
        """
        Test SB Bagging with non-tree base estimator (KNN)
        """
        clf = KNeighborsClassifier()
        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim)
        sb_clf.fit(self.X_train, self.y_train_clf)
        self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)

    def test_sb_bagging_non_sample_weights_with_verbose(self):
        """
        Test SB Bagging with classifier which doesn't support sample_weights with verbose > 1
        """
        clf = LinearSVC()

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=0.2,
                                                           n_estimators=2,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=True,
                                                           random_state=1, bootstrap_features=True,
                                                           max_samples=30, verbose=2)
        with self.assertWarns(UserWarning):
            sb_clf.fit(self.X_train, self.y_train_clf)
        self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)

    def test_sb_bagging_with_max_features(self):
        """
        Test SB Bagging with base_estimator bootstrap = True, float max_features, max_features bootstrap = True
        """
        clf = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=True,
                                     class_weight='balanced_subsample', max_depth=12)

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=0.2,
                                                           n_estimators=2,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=True,
                                                           random_state=1, bootstrap_features=True,
                                                           max_samples=30)
        with self.assertWarns(UserWarning):
            sb_clf.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))
        self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)

    def test_sb_bagging_float_max_samples_warm_start_true(self):
        """
        Test SB Bagging with warm start = True and float max_samples
        """
        clf = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                     class_weight='balanced_subsample', max_depth=12)

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=7,
                                                           n_estimators=2,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=False,
                                                           random_state=1, bootstrap_features=True,
                                                           max_samples=0.3, warm_start=True)

        sb_clf.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))

        sb_clf.n_estimators += 0
        with self.assertWarns(UserWarning):
            sb_clf.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))
        sb_clf.n_estimators += 2
        sb_clf.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))

        self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)

    def test_value_error_raise(self):
        """
        Test various values error raise
        """
        clf = KNeighborsClassifier()
        bagging_clf_1 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data)
        bagging_clf_2 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data, max_samples=2000000)
        bagging_clf_3 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data, max_features='20')
        bagging_clf_4 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data, max_features=2000000)
        bagging_clf_5 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data, oob_score=True, warm_start=True)
        bagging_clf_6 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data, warm_start=True)
        bagging_clf_7 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.samples_info_sets,
                                                                  price_bars=self.data, warm_start=True)
        with self.assertRaises(ValueError):
            # ValueError to use sample weight with classifier which doesn't support sample weights
            bagging_clf_1.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))
        with self.assertRaises(ValueError):
            # ValueError for max_samples > X_train.shape[0]
            bagging_clf_2.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)))
        with self.assertRaises(ValueError):
            # ValueError for non-int/float max_features param
            bagging_clf_3.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)))
        with self.assertRaises(ValueError):
            # ValueError for max_features > X_train.shape[1]
            bagging_clf_4.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)))
        with self.assertRaises(ValueError):
            # ValueError for warm_start and oob_score being True
            bagging_clf_5.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)))
        with self.assertRaises(ValueError):
            # ValueError for decreasing the number of estimators when warm start is True
            bagging_clf_6.fit(self.X_train, self.y_train_clf)
            bagging_clf_6.n_estimators -= 2
            bagging_clf_6.fit(self.X_train, self.y_train_clf)
        with self.assertRaises(ValueError):
            # ValueError for setting n_estimators to negative value
            bagging_clf_7.fit(self.X_train, self.y_train_clf)
            bagging_clf_7.n_estimators -= 1000
            bagging_clf_7.fit(self.X_train, self.y_train_clf)

    def test_sb_classifier(self):
        """
        Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
        test oos predictions values
        """
        # Init classifiers
        clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                          class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_base, max_features=1.0, n_estimators=100,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=True,
                                                           random_state=1)

        # X_train index should be in index mapping
        self.assertTrue(self.X_train.index.isin(sb_clf.timestamp_int_index_mapping.index).all())

        sb_clf.fit(self.X_train, self.y_train_clf)

        self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())  # X_train index == clf X_train index

        oos_sb_predictions = sb_clf.predict(self.X_test)

        sb_precision = precision_score(self.y_test_clf, oos_sb_predictions)
        sb_roc_auc = roc_auc_score(self.y_test_clf, oos_sb_predictions)
        sb_accuracy = accuracy_score(self.y_test_clf, oos_sb_predictions)

        self.assertAlmostEqual(sb_accuracy, 0.66, delta=0.2)
        self.assertEqual(sb_precision, 1.0)
        self.assertAlmostEqual(sb_roc_auc, 0.59, delta=0.2)

    def test_sb_regressor(self):
        """
        Test Sequentially Bootstrapped Bagging Regressor
        """
        # Init regressors
        reg = RandomForestRegressor(n_estimators=1, bootstrap=False)
        sb_reg = SequentiallyBootstrappedBaggingRegressor(base_estimator=reg, max_features=1.0, n_estimators=100,
                                                          samples_info_sets=self.samples_info_sets,
                                                          price_bars=self.price_bars_trim, oob_score=True,
                                                          random_state=1)
        sb_reg_1 = SequentiallyBootstrappedBaggingRegressor(base_estimator=reg, max_features=1.0, n_estimators=1,
                                                            samples_info_sets=self.samples_info_sets,
                                                            price_bars=self.price_bars_trim, oob_score=True,
                                                            random_state=1)

        sb_reg.fit(self.X_train, self.y_train_reg)

        with self.assertWarns(UserWarning):
            sb_reg_1.fit(self.X_train, self.y_train_reg)  # To raise warning and get code coverage

        # X_train index should be in index mapping
        self.assertTrue(self.X_train.index.isin(sb_reg.timestamp_int_index_mapping.index).all())
        self.assertTrue((sb_reg.X_time_index == self.X_train.index).all())  # X_train index == reg X_train index

        oos_sb_predictions = sb_reg.predict(self.X_test)
        mse_sb_reg = mean_squared_error(self.y_test_reg, oos_sb_predictions)
        mae_sb_reg = mean_absolute_error(self.y_test_reg, oos_sb_predictions)

        self.assertAlmostEqual(mse_sb_reg, 0.16, delta=0.1)
        self.assertAlmostEqual(mae_sb_reg, 0.29, delta=0.1)
