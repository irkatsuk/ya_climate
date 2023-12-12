import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, jarque_bera, shapiro


def check_data(data_df):
    pd.set_option('display.max_columns', None)

    print('\033[1m' + 'Изучим исходные данные' + '\033[0m')
    print(data_df.info())
    print(data_df.shape)

    missed_cells = data_df.isnull().sum().sum() / (data_df.shape[0] *
                                                   (data_df.shape[1] - 1))
    missed_rows = sum(data_df.isnull().sum(axis=1) > 0) / data_df.shape[0]
    print('\033[1m' + "\nПроверка пропусков" + '\033[0m')
    print("Количество пропусков: {:.0f}".format(data_df.isnull().sum().sum()))
    # print("Количество пропусков2: \n", data_df.isna().sum())
    print("Доля пропусков {:.1%}".format(missed_cells) + '\033[0m')
    print("Доля строк, содержащих пропуски {:.1%}".format(missed_rows))

    # duplicates
    print('\033[1m' + "\nПроверка на дупликаты" + '\033[0m')
    print('Количество полных дупликатов: ', data_df.duplicated().sum())
    duplicateRows = data_df[data_df.duplicated()]
    print(duplicateRows)

    # data
    print('\033[1m' + "\nПервые 5 строчек датасета" + '\033[0m')
    print(data_df.head())  # tail(7)

    print('\033[1m' + '\nОписание количественных данных:' + '\033[0m')
    print(data_df.describe().T)

    print('\033[1m' + '\nОписание категориальных данных:' + '\033[0m')
    print(data_df.describe(include='object').T)

    print(
        '\033[1m' + '\nВывод уникальных значений по каждому категориальному признаку:' + '\033[0m')
    df_object = data_df.select_dtypes(include='object').columns

    for i in df_object:
        print('\033[1m' + '_' + str(i) + '\033[0m')
        print(data_df[i].value_counts())


def plot_hist(data, col_column, filename='project.png'):
    '''
    Функция отрисовки гистограмм и ящика с усами для количесвтенных переменных.
    На вход: исходная таблица и список количественных переменных.
    На выходе: графики
    '''
    rows = len(col_column)
    f, ax = plt.subplots(rows, 2, figsize=(8, 15))
    f.tight_layout()
    f.set_figheight(30)
    f.set_figwidth(14)
    plt.rcParams.update({'font.size': 18})

    for i, col in enumerate(col_column):
        sns.histplot(data[col], kde=True, bins=24, ax=ax[i, 0])
        sns.boxplot(data[col], ax=ax[i, 1])

        ax[i, 0].set_xlabel(col)
        ax[i, 1].set_xlabel(col)
        ax[i, 0].set_ylabel('Количество')
    plt.suptitle("Гистограмма и ящик с усами для количесвтенных данных",
                 fontsize=22, y=1.01)
    plt.show()
    # f.savefig(filename)


def cat_graph(df, cat_feat, filename='project_cat.png'):
    '''
    Функция отрисовки круговых диаграмм для категориальных переменных.
    На вход: исходная таблица и список категориальных переменных.
    На выходе: графики
    '''

    cols = 2
    rows = int(np.ceil(len(cat_feat) / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
    plt.tight_layout()

    count = -1
    for i in range(rows):
        for x in range(cols):
            count += 1
            col = cat_feat[count]
            df1 = pd.DataFrame(df.groupby([col])[col].count())
            axs[i, x].pie(x=df1[col],
                          labels=df1.index,
                          autopct='%1.1f%%', )
            axs[i, x].title.set_text(str(col))

    plt.suptitle('Круговые диаграммы категориальных признаков', fontsize=20,
                 y=1.05)

    plt.show()
    # fig.savefig(filename)


def normal_sum_test(x, Ptest):
    """
    Функция проверяет нормальность/ненормальность распределения
    по сумме 3-х тестов: Шапиро, Андерсона-Дарлинга, Харке-Бера

    На выходе:
    1 - ненормальное распределение
    0 - нормальное распредление

    Принцип большинства заложен.
    Внутри есть функция расчёта критерия Андерсона, исходя из уровня значимсоти
    """

    def anderson_chois_sig(A, Ptest):
        if Ptest == 0.05:
            ander = A[2]
        elif Ptest == 0.01:
            ander = A[4]
        return ander

    def normalnost_anderson(x, Ptest):
        A2, crit, sig = anderson(x, dist='norm')
        ad_pass = (A2 < crit)
        norm = anderson_chois_sig(ad_pass, Ptest)
        if norm == False:
            return 1
        return 0

    # print(x)
    p_shapiro = shapiro(x)[1]
    p_jarque = jarque_bera(x)[1]

    if p_shapiro < Ptest:
        p_shapiros = 1
    else:
        p_shapiros = 0

    if p_jarque < Ptest:
        p_jarques = 1
    else:
        p_jarques = 0

    p_anderson = normalnost_anderson(x,
                                     Ptest)  # 1 - ненормальное, 0 - нормальное

    p_sum = p_shapiros + p_anderson + p_jarques

    if (p_sum > 1):
        return 1
    else:
        return 0
