from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "telco_raw.csv"
OUTPUT_PATH = BASE_DIR / "telco_preprocessing.csv"


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


def save_data(df, path):
    df.to_csv(
        path,
        index=False
    )


def main():

    print(f"Loading data from: {DATA_PATH}")

    df = load_data(DATA_PATH)

    df = preprocess(df)

    save_data(df, OUTPUT_PATH)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print("Preprocessing selesai")


if __name__ == "__main__":
    main()