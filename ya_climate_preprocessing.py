import ya_climate_common


def convert_to_pep(df):
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()
    return df


def process_feelling(x):
    if x < 0:
        return "неизвестно"
    if x == 0:
        return "неприемлемо"
    return "приемлемо"


def process_comfort(x):
    if x < 1:
        return "неизвестно"
    if x < 2:
        return "очень неудобно"
    if x < 3:
        return "неудобно"
    if x <= 4:
        return "нормально"
    if x <= 5:
        return "комфортно"
    return "очень комфортно"


def process_closing(x):
    if x < 0:
        return "неизвестно"
    if x == 0:
        return "открыто"
    return "закрыто"


def process_connected(x):
    if x < 0:
        return "неизвестно"
    if x == 0:
        return "выключен"
    return "включен"


def convert_to_celsius(x):
    celsius = (x - 32) * 5 / 9.0
    return celsius


def stage1(df):
    # конвертируем в pep8
    df = convert_to_pep(df)

    # типы столбцов
    df["год"] = df["год"].astype(int)

    # удаление строк с отсутствующим возрастом
    df = df.dropna(subset=["возраст"])
    df["возраст"] = df["возраст"].astype(int)

    col = "режим_при_смешанном_типе_охлаждения"
    df[col] = df[col].fillna("отсутствует")

    col = "способ_обогрева"
    df[col] = df[col].fillna("отсутствует")

    col = "пол"
    df[col] = df[col].fillna("не указан")

    col = "ощущение_движения_воздуха_(bool)"
    df[col] = df[col].fillna(-1.0)
    df[col] = df[col].apply(process_feelling)
    df[col] = df[col].astype(object)

    col = "оценка_комфорта"
    df[col] = df[col].fillna(df.groupby(["способ_охлаждения"])[col].
                             transform("mean"))
    df["оценка_комфорта_кат"] = df[col].apply(process_comfort)

    col = "температура_воздуха_на_улице"
    average_col = "среднемесячная_температура_на_улице"
    df[col] = df[col].fillna(df.groupby(["город", "время_года"])[average_col].
                             transform("mean"))

    df.drop(["рост", "вес"], axis=1, inplace=True)

    col = "занавески"
    df[col] = df[col].fillna(-1.0)
    df[col] = df[col].apply(process_closing)
    df[col] = df[col].astype(object)

    col = "вентилятор"
    df[col] = df[col].fillna(-1.0)
    df[col] = df[col].apply(process_connected)
    df[col] = df[col].astype(object)

    col = "окно"
    df[col] = df[col].fillna(-1.0)
    df[col] = df[col].apply(process_closing)
    df[col] = df[col].astype(object)

    col = "двери"
    df[col] = df[col].fillna(-1.0)
    df[col] = df[col].apply(process_closing)
    df[col] = df[col].astype(object)

    col = "отопление"
    df.loc[df["способ_охлаждения"] == "Кондиционирование", col] = 1.0
    df.loc[df["режим_при_смешанном_типе_охлаждения"] == "Кондиционирование",
           col] = 1.0
    df.loc[df["способ_охлаждения"] == "Вентиляция", col] = 0.0
    df.loc[df["режим_при_смешанном_типе_охлаждения"] == "Вентиляция",
           col] = 0.0
    df[col] = df[col].apply(lambda x: "выключено" if x == 0.0 else "включено")
    df[col] = df[col].astype(object)

    # удаляем дубликаты
    df = df.drop_duplicates()

    # удаляем выбросы
    new_name = {"Cубтропический океанический": "Cубтропический океанический",
                "Cубтроп океанич": "Cубтропический океанический",
                "Субтропическое высокогорье": "Субтропическое высокогорье",
                "Тропическая влажная саванна": "Тропическая влажная саванна",
                "Жаркий полузасушливый": "Жаркий полузасушливый",
                "Влажный субтропический муссонный":
                    "Влажный субтропический муссонный"}
    df["климат"] = df["климат"].map(new_name)

    new_name = {"Без изменений": "Без изменений",
                "Теплее": "Теплее",
                "Холоднее": "Холоднее",
                "Холодн": "Холоднее",
                "Тепле": "Теплее"}
    col = "предпочтительное_изменение_температуры"
    df[col] = df[col].map(new_name)

    # убираем выбросы с помощью тройного интерквартильного размаха
    col = "скорость_воздуха"
    q1 = df[col].quantile(0.25)
    q2 = df[col].quantile(0.75)
    delta = q2 - q1
    lower_bound = q1 - 3 * delta
    upper_bound = q2 + 3 * delta
    median = df[col].median()
    df.loc[(df[col] < lower_bound), col] = median
    df.loc[(df[col] > upper_bound), col] = median

    col = "температура_воздуха_в_помещении"
    df[col] = df[col].apply(lambda x: convert_to_celsius(x) if x > 50 else x)

    col = "среднемесячная_температура_на_улице"
    mean_temp = df.groupby("город")[col].mean()
    df.loc[df[col] > 100, col] = df["город"].map(mean_temp)
    df[col] = df[col].apply(lambda x: convert_to_celsius(x) if x > 50 else x)

    return df
