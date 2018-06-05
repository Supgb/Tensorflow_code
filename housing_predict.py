import pandas, hashlib, numpy as np, tarfile, os
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

"""
Preprocessing steps----------------------------------------------------------
"""

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
# DataFrame index axis
rooms_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pandas.read_csv(csv_path)


# RANDOM SPLIT train and test set
def test_set_check(id, test_radio, hash):
    return hash(np.int64(id)).digest()[-1] < 256 * test_radio


def split_train_test(data, test_radio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_radio,hash))
    return data[~in_test_set], data[in_test_set]


# STRAT SPLIT train and test set
def strat_split(data, test_size, index):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(data, data[index]):
        return data.loc[train_index], data.loc[test_index]


# For dropping text or numerical atrributes
def attribute_drop(dataset1, dataset2, attribute):
    for set in (dataset1, dataset2):
        set.drop([attribute], axis=1, inplace=True)


# TODO: fill missing value of train dataset
def data_clean(num_array, strategy="median"):
    imputer = Imputer(strategy=strategy)
    return imputer.fit_transform(num_array)


# TODO: Using ONE-HOT encoder on text and categorical attributes
def one_hot_encode(cat_array, sparse_output=False):
    encode = LabelBinarizer(sparse_output=sparse_output)
    return encode.fit_transform(cat_array)


# TODO: Custom transformers for attributes combination
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix] / X[:, household_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# TODO: Feature scaling
def feature_scaling_with_standardscaler(num_array):
    scaler = StandardScaler()
    return scaler.fit(num_array)


# TODO: Pipeline for transformation
# Handler of pandas' DataFrame for sklearn
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attrib_names):
        self.attrib_names = attrib_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attrib_names].values


def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation', scores.std())


if __name__ == "__main__":
    housing = load_housing_data()
    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    housing['income_cat'].where(housing['income_cat']<5, 5, inplace=True)
    train_set, test_set = strat_split(housing, 0.2, 'income_cat')
    housing = train_set.drop('median_house_value', axis=1)
    housing_labels = train_set['median_house_value']
    housing_num = housing.drop('ocean_proximity', axis=1)

    # Numerical pipeline
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])
    # Combine each other
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_perpared = full_pipeline.fit_transform(housing)

    """
    Training and evaluating steps----------------------------------------------------------
    """
    try:
        lin_reg = joblib.load('my_lin')
    except FileNotFoundError:
        lin_reg = LinearRegression()
        lin_reg.fit(housing_perpared, housing_labels)
        joblib.dump(lin_reg, 'my_lin')
    try:
        tree_reg = joblib.load('my_tree')
    except FileNotFoundError:
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_perpared, housing_labels)
        joblib.dump(tree_reg, 'my_tree')
    try:
        forest_reg = joblib.load('my_forest')
    except FileNotFoundError:
        forest_reg = RandomForestRegressor()
        forest_reg.fit(housing_perpared, housing_labels)
        joblib.dump(forest_reg, 'my_forest')

    """
    Select and train models
    """
    housing_predict = lin_reg.predict(housing_perpared)
    lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predict))
    lin_scores = cross_val_score(lin_reg, housing_perpared, housing_labels,
                                 scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)

    housing_predict = tree_reg.predict(housing_perpared)
    tree_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predict))
    tree_scores = cross_val_score(tree_reg, housing_perpared, housing_labels,
                                  scoring='neg_mean_squared_error', cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)

    housing_predict = forest_reg.predict(housing_perpared)
    forest_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predict))
    forest_scores = cross_val_score(forest_reg, housing_perpared, housing_labels,
                                    scoring='neg_mean_squared_error', cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)

    """
    Fine-tune selected models
    """
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing_perpared, housing_labels)