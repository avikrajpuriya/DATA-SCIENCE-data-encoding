# Data Enc – Data Preprocessing: Skewness, Scaling and Encoding

## Dataset
`urban_climate.csv` is the urban-climate multi-attribute dataset used for this experiment. It contains numerical and categorical attributes (11,040 rows and 20 columns).

## Requirements
- Python 3.10 or later
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter (only needed for the notebook)

Install dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Files
- `data_enc.py` – standalone Python implementation.
- `Data_Enc_Preprocessing.ipynb` – Jupyter notebook with explanations, code and outputs when run.
- `Data_Enc_Preprocessing.pdf` – printable experiment report.
- `urban_climate.csv` – input dataset.
- `README.md` – project instructions.

## Run the Python script
Keep `data_enc.py` and `urban_climate.csv` in the same folder, then run:

```bash
python data_enc.py
```

The script creates:
- `urban_climate_minmax_scaled.csv`
- `urban_climate_preprocessed.csv`

## Experiment coverage
1. Loads and inspects the multi-attribute dataset.
2. Calculates skewness and plots histograms with KDE curves for numerical attributes.
3. Implements Min-Max scaling manually and with scikit-learn's `MinMaxScaler`.
4. Demonstrates ordinal encoding using a derived precipitation-level feature: Low = 0, Moderate = 1, High = 2.
5. Applies one-hot encoding to nominal categorical features using `pandas.get_dummies(drop_first=True)`.
6. Displays original and transformed data and verifies normalized minimum/maximum and standardized mean/standard deviation.

## Important dataset note
The supplied urban-climate dataset does not contain an existing education-level or other explicitly ordered categorical feature. To demonstrate ordinal encoding without pretending such a column exists, the notebook derives `precipitation_level` from precipitation quantiles and maps Low, Moderate and High to 0, 1 and 2. This is a derived demonstration feature.

## Scaling note
- Min-Max scaling maps each selected feature to the range [0, 1].
- Standardization maps numerical features to approximately mean 0 and population standard deviation 1. The verification uses `ddof=0`, matching scikit-learn's `StandardScaler`.

## GitHub
Extract this ZIP and upload the folder contents to your GitHub repository. Do not upload generated CSV outputs unless your instructor asks for them; they are recreated when the script or notebook is run.
