"""
CI-friendly preprocessing that creates sample data if raw .psv files unavailable.
For full preprocessing, ensure .psv files are in the data_dir.
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]
DEMO_COLS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
CLINICAL_COLS = VITAL_COLS + LAB_COLS
SEED = 42


def load_all_patients(data_dir: str) -> pd.DataFrame:
    """Load all patient .psv files, or create synthetic data if none found."""
    data_dir = os.path.expanduser(data_dir)
    data_dir = os.path.abspath(data_dir)

    # Try to find real .psv files
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.psv')))
    if not all_files and os.path.isdir(data_dir):
        all_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.psv'), recursive=True))

    if all_files:
        log.info(f"Found {len(all_files):,} .psv files in {data_dir}")
        dfs = []
        for i, filepath in enumerate(all_files):
            patient_id = os.path.basename(filepath).replace('.psv', '')
            df = pd.read_csv(filepath, sep='|')
            df.insert(0, 'patient_id', patient_id)
            dfs.append(df)
            if (i + 1) % 5000 == 0:
                log.info(f"  Loaded {i+1:,}/{len(all_files):,} ...")
        combined = pd.concat(dfs, ignore_index=True)
        log.info(f"Merge complete → {combined.shape[0]:,} rows")
        return combined
    else:
        # Create synthetic data for testing (common in CI/testing scenarios)
        log.warning(f"No .psv files found in {data_dir}")
        log.info("Creating synthetic sample data for testing...")
        return create_synthetic_data(n_patients=1000)


def create_synthetic_data(n_patients: int = 1000) -> pd.DataFrame:
    """Generate synthetic sepsis data for testing without raw files."""
    np.random.seed(SEED)
    rows = []

    for p_id in range(1, n_patients + 1):
        n_obs = np.random.randint(10, 100)
        patient_id = f"p{p_id:06d}"
        sepsis_label = np.random.choice([0, 1], p=[0.6, 0.4])

        for t in range(n_obs):
            row = {
                'patient_id': patient_id,
                'ICULOS': t,
                'SepsisLabel': sepsis_label
            }

            for col in DEMO_COLS:
                if col == 'Age':
                    row[col] = np.random.randint(18, 90)
                elif col == 'Gender':
                    row[col] = np.random.choice([0, 1])
                else:
                    row[col] = np.random.choice([0, 1]) if col in ['Unit1', 'Unit2'] else 0

            for col in VITAL_COLS:
                if sepsis_label == 1:
                    row[col] = np.random.normal(100, 15)  # Sepsis often has abnormal vitals
                else:
                    row[col] = np.random.normal(100, 10)  # Normal range

            for col in LAB_COLS:
                if sepsis_label == 1:
                    row[col] = np.random.normal(50, 20)
                else:
                    row[col] = np.random.normal(50, 15)

            rows.append(row)

    log.info(f"Generated {len(rows)} synthetic observations from {n_patients} patients")
    return pd.DataFrame(rows)


def aggregate_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time-series data into per-patient feature vectors."""
    log.info("Aggregating features per patient...")
    rows = []
    groups = list(df.groupby('patient_id'))

    for i, (pid, group) in enumerate(groups):
        row = {'patient_id': pid}
        row['SepsisLabel'] = int(group['SepsisLabel'].max())

        for col in DEMO_COLS:
            row[col] = group[col].iloc[0] if col in group.columns else 0

        row['ICULOS_max'] = group['ICULOS'].max() if 'ICULOS' in group.columns else 0
        row['n_observations'] = len(group)

        for col in CLINICAL_COLS:
            series = group[col].dropna() if col in group.columns else pd.Series([])
            if len(series) > 0:
                row[f'{col}_mean'] = series.mean()
                row[f'{col}_std'] = series.std() if len(series) > 1 else 0.0
                row[f'{col}_min'] = series.min()
                row[f'{col}_max'] = series.max()
                row[f'{col}_last'] = series.iloc[-1]
            else:
                for s in ['mean', 'std', 'min', 'max', 'last']:
                    row[f'{col}_{s}'] = np.nan
            missing_rate = group[col].isnull().mean() if col in group.columns else 1.0
            row[f'{col}_missing_rate'] = missing_rate

        for col in VITAL_COLS:
            series = group[col].dropna() if col in group.columns else pd.Series([])
            if len(series) >= 3:
                row[f'{col}_trend'] = (series.iloc[-1] - series.iloc[0]) / len(series)
            else:
                row[f'{col}_trend'] = 0.0

        rows.append(row)
        if (i + 1) % 500 == 0:
            log.info(f"  Progress: {i+1:,}/{len(groups):,}")

    result = pd.DataFrame(rows)
    log.info(f"Aggregation complete → shape: {result.shape}")
    return result


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with median."""
    log.info(f"Missing values before imputation: {X.isnull().sum().sum():,}")
    X_imputed = X.fillna(X.median())
    log.info(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
    return X_imputed


def remove_low_quality_features(X: pd.DataFrame, corr_threshold: float = 0.98) -> pd.DataFrame:
    """Remove zero-variance and highly-correlated features."""
    # Zero variance
    zero_var = X.columns[X.std() == 0].tolist()
    if zero_var:
        log.info(f"Removing {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)

    # High correlation
    corr_mat = X.corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    high_corr = [c for c in upper_tri.columns if any(upper_tri[c] > corr_threshold)]
    if high_corr:
        log.info(f"Removing {len(high_corr)} highly-correlated features (>{corr_threshold})")
        X = X.drop(columns=high_corr)

    log.info(f"Shape after cleaning: {X.shape}")
    return X


def split_and_balance(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Train-test split with SMOTE balancing and scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    log.info(f"Split → train: {X_train.shape}, test: {X_test.shape}")

    log.info("Applying SMOTE to training set...")
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info(f"  Post-SMOTE → train: {X_train_res.shape} | "
             f"Sepsis: {y_train_res.sum():,} | Non-Sepsis: {(y_train_res==0).sum():,}")

    log.info("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_sc, X_test_sc, y_train_res, y_test


def save_outputs(X_train, X_test, y_train, y_test, X_full, y_full,
                 output_dir: str = './sepsis_preprocessing'):
    """Save train/test/full datasets to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['SepsisLabel'] = y_train.values
    test_df = X_test.copy()
    test_df['SepsisLabel'] = y_test.values
    full_df = X_full.copy()
    full_df['SepsisLabel'] = y_full.values

    train_path = os.path.join(output_dir, 'sepsis_preprocessing_train.csv')
    test_path = os.path.join(output_dir, 'sepsis_preprocessing_test.csv')
    full_path = os.path.join(output_dir, 'sepsis_preprocessing.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    full_df.to_csv(full_path, index=False)

    log.info(f"Saved:")
    log.info(f"  {train_path} → {train_df.shape}")
    log.info(f"  {test_path}  → {test_df.shape}")
    log.info(f"  {full_path}  → {full_df.shape}")


def preprocess(data_dir: str, output_dir: str = './sepsis_preprocessing'):
    """Main preprocessing pipeline."""
    log.info("=" * 70)
    log.info("  SEPSIS ICU PREPROCESSING PIPELINE")
    log.info("=" * 70)

    # 1. Load
    df_raw = load_all_patients(data_dir)

    # 2. Feature engineering
    df_patient = aggregate_patient_features(df_raw)

    # 3. Extract features and labels
    exclude_cols = ['patient_id', 'SepsisLabel']
    feature_cols = [c for c in df_patient.columns if c not in exclude_cols]
    X = df_patient[feature_cols].copy()
    y = df_patient['SepsisLabel'].copy()

    # 4. Clean
    X = handle_missing_values(X)
    X = remove_low_quality_features(X)

    # 5. Split + SMOTE + Scale
    X_train, X_test, y_train, y_test = split_and_balance(X, y)

    # 6. Save
    save_outputs(X_train, X_test, y_train, y_test, X, y, output_dir)

    log.info("=" * 70)
    log.info("  PIPELINE COMPLETE — Data ready for model training!")
    log.info("=" * 70)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CI-friendly Sepsis ICU Preprocessing Pipeline'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./training_data',
        help='Path to folder with .psv files (creates synthetic data if not found)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./sepsis_preprocessing',
        help='Output folder for saving CSV files'
    )
    args = parser.parse_args()

    preprocess(data_dir=args.data_dir, output_dir=args.output_dir)
