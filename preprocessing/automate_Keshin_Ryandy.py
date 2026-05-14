"""
Automated preprocessing script for Titanic dataset.
Author: Keshin Ryandy

Struktur yang disarankan:
Eksperimen_SML_Keshin_Ryandy/
├── titanic_raw/
│   ├── train.csv
│   └── test.csv
└── preprocessing/
    ├── Eksperimen_MSML_Keshin_Ryandy.ipynb
    ├── automate_Keshin_Ryandy.py
    └── titanic_preprocessing/
        ├── train_preprocessed.csv
        └── test_preprocessed.csv

Cara menjalankan:
python preprocessing/automate_Keshin_Ryandy.py
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DIR = PROJECT_DIR / "titanic_raw"
OUTPUT_DIR = SCRIPT_DIR / "titanic_preprocessing"

TRAIN_FILE = RAW_DIR / "train.csv"
TEST_FILE = RAW_DIR / "test.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# FUNCTIONS
# =========================
def load_data(train_path: Path, test_path: Path):
    """Load raw train and test dataset."""
    if not train_path.exists():
        raise FileNotFoundError(f"File train tidak ditemukan: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"File test tidak ditemukan: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logging.info("Raw train shape: %s", train_df.shape)
    logging.info("Raw test shape : %s", test_df.shape)

    return train_df, test_df


def extract_and_encode_title(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Extract title from Name column and encode it."""
    combine = [train_df, test_df]

    for dataset in combine:
        dataset["Title"] = dataset["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

        dataset["Title"] = dataset["Title"].replace(
            [
                "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
                "Rev", "Sir", "Jonkheer", "Dona"
            ],
            "Rare"
        )

        dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
        dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
        dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

    title_mapping = {
        "Mr": 1,
        "Miss": 2,
        "Mrs": 3,
        "Master": 4,
        "Rare": 5
    }

    for dataset in combine:
        dataset["Title"] = dataset["Title"].map(title_mapping)
        dataset["Title"] = dataset["Title"].fillna(0).astype(int)

    logging.info("Title extraction and encoding completed.")
    return train_df, test_df


def drop_unused_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Drop columns that are not used for modeling."""
    train_df = train_df.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    test_df = test_df.drop(["Ticket", "Cabin", "Name"], axis=1)

    logging.info("Unused columns dropped.")
    return train_df, test_df


def encode_sex(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Encode Sex column: female = 1, male = 0."""
    for dataset in [train_df, test_df]:
        dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)

    logging.info("Sex encoding completed.")
    return train_df, test_df


def fill_age_by_sex_pclass(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fill missing Age using median Age based on Sex and Pclass group.
    This follows the EDA notebook logic.
    """
    for dataset in [train_df, test_df]:
        guess_ages = np.zeros((2, 3))

        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[
                    (dataset["Sex"] == i) &
                    (dataset["Pclass"] == j + 1)
                ]["Age"].dropna()

                age_guess = guess_df.median()

                if pd.isna(age_guess):
                    age_guess = dataset["Age"].dropna().median()

                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[
                    (dataset["Age"].isnull()) &
                    (dataset["Sex"] == i) &
                    (dataset["Pclass"] == j + 1),
                    "Age"
                ] = guess_ages[i, j]

        dataset["Age"] = dataset["Age"].astype(int)

    logging.info("Missing Age values handled.")
    return train_df, test_df


def create_age_band(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Convert Age into ordinal age bands."""
    for dataset in [train_df, test_df]:
        dataset.loc[dataset["Age"] <= 16, "Age"] = 0
        dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
        dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
        dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
        dataset.loc[dataset["Age"] > 64, "Age"] = 4

        dataset["Age"] = dataset["Age"].astype(int)

    logging.info("Age banding completed.")
    return train_df, test_df


def create_family_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Create FamilySize and IsAlone features, then drop SibSp, Parch, and FamilySize."""
    for dataset in [train_df, test_df]:
        dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
        dataset["IsAlone"] = 0
        dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

    train_df = train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
    test_df = test_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)

    logging.info("Family features completed.")
    return train_df, test_df


def create_interaction_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Create Age*Class interaction feature."""
    for dataset in [train_df, test_df]:
        dataset["Age*Class"] = dataset["Age"] * dataset["Pclass"]

    logging.info("Interaction feature Age*Class completed.")
    return train_df, test_df


def fill_and_encode_embarked(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fill missing Embarked using mode from train data and encode it."""
    freq_port = train_df["Embarked"].dropna().mode()[0]

    for dataset in [train_df, test_df]:
        dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
        dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    logging.info("Embarked missing value handling and encoding completed.")
    return train_df, test_df


def fill_and_band_fare(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fill missing Fare and convert Fare into ordinal fare bands."""
    # Mengikuti notebook: missing Fare hanya ada di test dan diisi median test.
    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].dropna().median())

    for dataset in [train_df, test_df]:
        dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
        dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
        dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
        dataset.loc[dataset["Fare"] > 31, "Fare"] = 3

        dataset["Fare"] = dataset["Fare"].astype(int)

    logging.info("Fare missing value handling and banding completed.")
    return train_df, test_df


def validate_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Validate processed data before saving."""
    if train_df.isnull().sum().sum() != 0:
        raise ValueError("Train dataset masih memiliki missing values.")

    if test_df.isnull().sum().sum() != 0:
        raise ValueError("Test dataset masih memiliki missing values.")

    logging.info("Processed train shape: %s", train_df.shape)
    logging.info("Processed test shape : %s", test_df.shape)
    logging.info("Train columns: %s", list(train_df.columns))
    logging.info("Test columns : %s", list(test_df.columns))


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
    """Save preprocessed train and test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / "train_preprocessed.csv"
    test_output = output_dir / "test_preprocessed.csv"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    logging.info("Processed train saved to: %s", train_output)
    logging.info("Processed test saved to : %s", test_output)


def preprocessing_pipeline():
    """Run full automated preprocessing pipeline."""
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    train_df, test_df = extract_and_encode_title(train_df, test_df)
    train_df, test_df = drop_unused_columns(train_df, test_df)
    train_df, test_df = encode_sex(train_df, test_df)
    train_df, test_df = fill_age_by_sex_pclass(train_df, test_df)
    train_df, test_df = create_age_band(train_df, test_df)
    train_df, test_df = create_family_features(train_df, test_df)
    train_df, test_df = create_interaction_features(train_df, test_df)
    train_df, test_df = fill_and_encode_embarked(train_df, test_df)
    train_df, test_df = fill_and_band_fare(train_df, test_df)

    validate_processed_data(train_df, test_df)
    save_processed_data(train_df, test_df, OUTPUT_DIR)

    logging.info("Automated preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    preprocessing_pipeline()
