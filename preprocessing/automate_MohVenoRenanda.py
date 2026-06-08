import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):

    # drop column
    df = df.drop(columns=["customerID"])

    # convert total charges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # duplicate
    df = df.drop_duplicates()

    # target encoding
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # categorical encoding
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # scaling
    scaler = StandardScaler()

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[numeric_cols] = scaler.fit_transform(
        df[numeric_cols]
    )

    return df


def save_data(df):
    df.to_csv(
        "telco_preprocessing.csv",
        index=False
    )


def main():

    df = load_data(
        "../telco_raw.csv"
    )

    df = preprocess(df)

    save_data(df)

    print("Preprocessing selesai")


if __name__ == "__main__":
    main()