from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ya_climate_common import get_ohe


def ya_regression(df):
    numerical_columns = ["температура_воздуха_на_улице",
                         "скорость_воздуха", "утепление", "rh",
                         "ощущение_температуры", "оценка_комфорта"]
    categorical_columns = ["способ_охлаждения", "отопление", "вентилятор",
                           "занавески", "климат", "способ_обогрева",
                           "предпочтительное_изменение_температуры"]
    x = df[numerical_columns + categorical_columns]
    y = df[['температура_воздуха_в_помещении']]

    SIZE = 0.15
    RANDOM_STATE = 42

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=SIZE, random_state=RANDOM_STATE)

    ohe = OneHotEncoder(sparse=False, drop='first')
    ohe.fit(x_train[categorical_columns])
    x_train_new = get_ohe(x_train, categorical_columns, ohe)
    x_test_new = get_ohe(x_test, categorical_columns, ohe)

    regression = LinearRegression()
    regression.fit(x_train_new, y_train)
    prediction = regression.predict(x_test_new)

    r2 = r2_score(y_test, prediction.flatten())
    mape = mean_absolute_percentage_error(y_test, prediction.flatten())
    mape = mape * 100
    print(f"Результат работы модели: r2 = {round(r2, 2)},"
          f"MAPE = {round(mape, 2)}")
