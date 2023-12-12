import math
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from ya_climate_common import normal_sum_test

def count_of_adv(x):
    if x <= 1:
        return "мало"
    elif x == 2:
        return "средне"
    return "много"


def make_age_cat(x):
    if x <= 44:
        return 'молодой возраст'
    elif x <= 59:
        return 'средний возраст'
    return 'пожилой возраст'


def standart_rh(x):
    if x < 40:
        return "Ниже нормы"
    elif 40 <= x <= 60:
        return "Норма"
    return "Выше нормы"


def corr(df, x, y):  # функция для расчета корреляции между двумя факторами
    if is_numeric_dtype(df[x]) and is_numeric_dtype(df[y]):
        if normal_sum_test(df[x], 0.05) == 1 and normal_sum_test(df[y],
                                                                 0.05) == 1:
            return round(pearsonr(df[x], df[y])[0], 3)
        return round(spearmanr(df[x], df[y])[0], 3)
    elif is_string_dtype(df[x]) and is_string_dtype(df[y]):
        df1 = df[[x, y]]
        tab = pd.crosstab(df1[x], df1[y])
        return round(math.sqrt(chi2_contingency(tab).statistic / (
                df1.shape[0] * (min(df1.shape) - 1))), 3)
    elif is_string_dtype(df[y]):
        x, y = y, x
    # расчет корреляционного значения Eta
    cat = df[x].unique()
    in_group = 0
    inter_group = 0
    for i in cat:
        df1 = df[df[x] == i]
        m = df1[y].mean()
        in_group += df1[y].apply(lambda x: (x - m) ** 2).sum()
        inter_group += df1[y].count() * (m - df[y].mean()) ** 2

    return round(inter_group / (in_group + inter_group), 3)


def cheddok(coef, x, y):
    # для оценки силы связи будем использовать шкалу Чеддока
    if 0.5 <= abs(coef) <= 0.7:
        return f"Заметная связь между факторами {x} и {y}. Коэффициент корреляции = {coef}"
    elif 0.7 < abs(coef) <= 0.9:
        return f"Сильная связь между факторами {x} и {y}. Коэффициент корреляции = {coef}"
    elif 0.9 < abs(coef) <= 1:
        return f"Очень сильная связь между факторами {x} и {y}. Коэффициент корреляции = {coef}"


def research(df):
    col = "количество_рекламаций_кат"
    df[col] = df["количество_рекламаций"].apply(count_of_adv)

    col = 'возрастная_группа'
    df[col] = df['возраст'].apply(make_age_cat)

    # средний возраст по полу и стране
    avg_age = pd.DataFrame(df.groupby(["пол", "страна"])["возраст"]
                           .mean().round().astype(int)).rename(
        columns=
        {"возраст": "средний_возраст"})

    # медианное значение температуры и влажности для каждого типа охлаждения
    med_type_of_cool = pd.DataFrame(df.groupby("способ_охлаждения")
                                    [["температура_воздуха_в_помещении", "rh"]]
                                    .median()).rename(
        columns=
        {"температура_воздуха_в_помещении": "медианная_температура",
         "rh": "медианная_влажность"})

    # средняя комфортная температура в зависимости от возрастной категории
    avg_comf_temp_age = pd.DataFrame(
        df[df["предпочтительное_изменение_температуры"]
           == "Без изменений"].groupby("возрастная_группа")
        ["температура_воздуха_в_помещении"]
            .apply(lambda x: round(x.sum() / x.count(), 1))).rename(
        columns={"температура_воздуха_в_помещении":
                     "средняя_комфортная_температура"})

    # процент удовлетворенных температурой респондентов по стране и полу
    df["удовлетворенность_температурой_%"] = \
        df.groupby(
            ["страна", "пол"])[
            "предпочтительное_изменение_температуры"].transform(
            lambda x: round(x.value_counts(normalize=True)[0] * 100, 1))

    '''
    сводная таблица, в которой данные сгруппированы по стране, полу,
    возрастной группе и посчитаны средняя температура воздуха в помещении,
    на улице и средняя относительная влажность для каждой из этих групп.
    '''
    avg_temp_rh = df.groupby(["страна", "пол", "возрастная_группа"])[
        ["температура_воздуха_в_помещении", "температура_воздуха_на_улице",
         "rh"]].apply(lambda x:
                      round(x.sum() / x.count(), 1)).rename(
        columns={
            "температура_воздуха_в_помещении": "средняя_теспература_воздуха_в_помещении",
            "температура_воздуха_на_улице": "средняя_температура_воздуха_на_улице",
            "rh": "средняя_относительная_влажность"
        }
    )
    # стандартная комфортная относительная влажность в офисах по СНиП: 40-60 %
    # создадим категориальный столбец для влажности
    df["rh_кат"] = df["rh"].apply(standart_rh)

    # исследуем корреляцию между параметрами
    list_of_corr = []  # список всех взаимосвязей, которые имеют значение
    for i in range(len(df.columns) - 1):
        for j in range(i + 1, len(df.columns)):
            coef = corr(df, df.columns[i], df.columns[j])
            if 1 >= abs(coef) >= 0.5:
                list_of_corr.append([coef, cheddok(coef, df.columns[i],
                                                  df.columns[
                    j])])
    return df, avg_age, med_type_of_cool, avg_comf_temp_age, avg_temp_rh, list_of_corr
