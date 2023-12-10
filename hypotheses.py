import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, kruskal, f_oneway, pearsonr, spearmanr

import ya_climate_common
from ya_climate_preprocessing import stage1

df = pd.read_csv("ya_climate.csv", sep=";", decimal=".", encoding='windows-1251')


def printing(string, p):
    if p < 0.05:
        print(string, "влияние фактора на отклик обнаружено")
    else:
        print(string, "влияние фактора на отклик не обнаружено",)


def make_old(x):
    if x <= 44:
        return 'молодой'
    elif x <= 59:
        return 'средний'
    return 'пожилой'


def hypotheses(dat):
    # Проверка каждой гипотезы имеет одно строение:
    # 1. Формирование групп
    # 2. Проверка их на нормальность
    # 3. Вывод результата проверки с учётом нормальности распределения данных

    # Проверка гипотезы о влиянии способа охлаждения на оценку комфорта
    grp1 = dat[dat.способ_охлаждения == 'Кондиционирование']['оценка_комфорта']
    grp2 = dat[dat.способ_охлаждения == 'Смешанный']['оценка_комфорта']
    grp3 = dat[dat.способ_охлаждения == 'Вентиляция']['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05 and shapiro(grp3)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии способа охлаждения на оценку комфорта:", f_oneway(grp1, grp2, grp3)[1])
    else:
        printing("Гипотеза о влиянии способа охлаждения на оценку комфорта:", kruskal(grp1, grp2, grp3)[1])

    # Проверка гипотезы о влиянии пола на оценку комфорта
    grp1 = dat[dat.пол == 'Мужской']['оценка_комфорта']
    grp2 = dat[dat.пол == 'Женский']['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии пола на оценку комфорта:", ttest_ind(grp1, grp2, equal_var=False)[1])
    else:
        printing("Гипотеза о влиянии пола на оценку комфорта:", mannwhitneyu(grp1, grp2)[1])

    # Проверка гипотезы о влиянии возрастной группы на оценку комфорта
    grp1 = dat[dat.возрастная_группа == 'молодой']['оценка_комфорта']
    grp2 = dat[dat.возрастная_группа == 'средний']['оценка_комфорта']
    grp3 = dat[dat.возрастная_группа == 'пожилой']['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05 and shapiro(grp3)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии возрастной группы на оценку комфорта:", f_oneway(grp1, grp2, grp3)[1])
    else:
        printing("Гипотеза о влиянии возрастной группы на оценку комфорта:", kruskal(grp1, grp2, grp3)[1])

    # Проверка гипотезы о взаимосвязи количества рекламаций и оценки комфорта
    grp1 = dat['оценка_комфорта']
    grp2 = dat['количество_рекламаций']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05
    if normal:
        printing("Гипотеза о взаимосвязи количества рекламаций и оценки комфорта:",
                 ttest_ind(grp1, grp2, equal_var=False)[1])
    else:
        printing("Гипотеза о взаимосвязи количества рекламаций и оценки комфорта:", mannwhitneyu(grp1, grp2)[1])

    # Проверка гипотезы о влиянии страны на оценку комфорта
    grp1 = dat[dat.страна == 'США']['оценка_комфорта']
    grp2 = dat[dat.страна == 'Австралия']['оценка_комфорта']
    grp3 = dat[dat.страна == 'Индия']['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05 and shapiro(grp3)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии страны на оценку комфорта:", f_oneway(grp1, grp2, grp3)[1])
    else:
        printing("Гипотеза о влиянии страны на оценку комфорта:", kruskal(grp1, grp2, grp3)[1])

    # Проверка гипотезы о влиянии влажности на оценку комфорта
    grp1 = dat['rh']
    grp2 = dat['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии влажности на оценку комфорта:", pearsonr(grp1, grp2)[1])
    else:
        printing("Гипотеза о влиянии влажности на оценку комфорта:", spearmanr(grp1, grp2)[1])

    # Проверка гипотезы о влиянии скорости воздуха на оценку комфорта
    grp1 = dat['скорость_воздуха']
    grp2 = dat['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии скорости воздуха на оценку комфорта:",
                 pearsonr(grp1, grp2)[1])
    else:
        printing("Гипотеза о влиянии скорости воздуха на оценку комфорта:", spearmanr(grp1, grp2)[1])

    # Проверка гипотезы о влиянии температуры воздуха в помещении на оценку комфорта
    grp1 = dat['температура_воздуха_в_помещении']
    grp2 = dat['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии температуры воздуха в помещении на оценку комфорта:",
                 pearsonr(grp1, grp2)[1])
    else:
        printing("Гипотеза о влиянии температуры воздуха в помещении на оценку комфорта:",
                 spearmanr(grp1, grp2)[1])

    # Проверка гипотезы о влиянии утепления на оценку комфорта
    grp1 = dat['утепление']
    grp2 = dat['оценка_комфорта']
    normal = shapiro(grp1)[1] >= 0.05 and shapiro(grp2)[1] >= 0.05
    if normal:
        printing("Гипотеза о влиянии тутепления на оценку комфорта:",
                 pearsonr(grp1, grp2)[1])
    else:
        printing("Гипотеза о влиянии утепления на улице на оценку комфорта:", spearmanr(grp1, grp2)[1])


df = stage1(df)
df['возрастная_группа'] = df['возраст'].apply(make_old)
hypotheses(df)
