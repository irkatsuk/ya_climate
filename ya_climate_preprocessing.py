import ya_climate_common


def convert_to_pep(df):
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()
    return df


def stage1(df):
    # конвертируем в pep8
    df = convert_to_pep(df)

    # типы столбцов
    df["год"] = df["год"].astype(int)

    # удаление строк с отсутствующим возрастом
    df = df.dropna(subset=["возраст"])
    df["возраст"] = df["возраст"].astype(int)
    return df
