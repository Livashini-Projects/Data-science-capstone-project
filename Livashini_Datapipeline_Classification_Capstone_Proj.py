import re
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def dataprep(df, Train, scaling_data=True, Handling_Outliers=True,
             Handling_Missing_Values_Not_DT=True, Handling_Missing_Values_DT=False,
             encoding=True, is_training_data=True):

    # 1. Duplicates & ID Columns
    ##############################################################
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    df.drop(columns=['incident_id'], inplace=True)
    # 4. Missing Value Imputation
    ##############################################################
    if Handling_Missing_Values_Not_DT:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        # cat_cols = df.select_dtypes(exclude='number').columns.tolist()

        # Categorical: fill with mode from Train, but in the crime dataset, the weapon_used column is different. The missing values are not random — they are meaningful. So instead of filling with the mode (which would bias the data toward the most frequent weapon like knife or gun), we use a separate category as no weapon
        
        for col in cat_cols:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
            replaceval = "No Weapon"
            df[col] = df[col].fillna(replaceval)

        # Numeric: MICE — fit on Train only to avoid leakage
        mice = IterativeImputer(max_iter=10, random_state=42, min_value=0)

        if is_training_data:
            mice.fit(Train[numeric_cols])
            joblib.dump(mice, 'mice_imputer_class.pkl')
        else:
            mice = joblib.load('mice_imputer_class.pkl')

        # create missing indicators column-by-column BEFORE imputation
        for col in numeric_cols:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
        df[numeric_cols] = mice.transform(df[numeric_cols])

    if Handling_Missing_Values_DT:
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        for col in df.select_dtypes(include='number').columns:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(9999999)
    
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("No Weapon")

    # 5. Outlier Handling (numeric columns only)
    ##############################################################
    if Handling_Outliers:
        # Iterate over df's numeric columns, excluding _missing indicators added in step 4
        # Outlier bounds (Q1, Q3, IQR) are always computed from Train to avoid leakage
        cols_to_check = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing')
        ]

        for col in cols_to_check:
            Q1 = Train[col].quantile(0.25)
            Q3 = Train[col].quantile(0.75)
            IQR = Q3 - Q1
            LOF = Q1 - 3 * IQR  # Lower Outer Fence
            UOF = Q3 + 3 * IQR  # Upper Outer Fence
            df[col + "_flag"] = df[col].apply(lambda x: 1 if x < LOF or x > UOF else 0)
            df[col] = df[col].clip(lower=LOF, upper=UOF)

    # 6. Feature Engineering & Log Transforms
    ##############################################################


    # investigation weight 
    df['investigation_load'] = df['investigation_duration_days'] * df['officers_assigned']

    # risk area
    df['risk_density'] = df['population_density_per_sqkm'] * df['prior_incidents_same_location']

    # weapon used flag
    df['weapon_used_flag'] = df['weapon_used'].apply(lambda x: 1 if x != "No Weapon" else 0)

    # high crime
    df['high_crime'] = (
    (df['crime_severity_score'] > df['crime_severity_score'].median()) & (df['victim_count'] > 0)).astype(int)
    
    # action by police ratio
    df['police_efficiency'] = np.where((df['response_time_minutes'].isna()) | (df['response_time_minutes'] == 0),0,df['crime_severity_score'] / (df['response_time_minutes'] + 1))

    # 7. Scaling (numeric columns only)
    ##############################################################
    if scaling_data:
        # Exclude binary indicator/flag columns — scaling adds no benefit
        # and makes the values harder to interpret (0/1 becomes e.g. -0.5 / 2.3)
        numeric_cols = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing') and not c.endswith('_flag')
        ]

        all_numeric_cols = numeric_cols
       
        scaler = RobustScaler()  # RobustScaler uses Median & IQR — not sensitive to outliers

        if is_training_data:
            df[all_numeric_cols] = scaler.fit_transform(df[all_numeric_cols])
            joblib.dump(scaler, 'scaler_class.pkl')
        else:
            scaler = joblib.load('scaler_class.pkl')
            df[all_numeric_cols] = scaler.transform(df[all_numeric_cols])

    # 8. Encoding (categorical columns only)
    ##############################################################
    if encoding:
        numeric_cols = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing') and not c.endswith('_flag')
        ]
        base_cat_cols = [
            c for c in df.select_dtypes(exclude='number').columns
            if not c.endswith('_missing') and not c.endswith('_flag')
        ]
        cat_cols = base_cat_cols
        all_numeric_cols = numeric_cols

        # OHE is fitted on df's categorical columns after all prior cleaning steps
        # This is correct as long as is_training_data=True is called with the full training set
        # handle_unknown='ignore' silently zero-encodes unseen categories at inference —
        # intentional: new categories contribute nothing to the model output
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if is_training_data:
            ohe_array = ohe.fit_transform(df[cat_cols])
            joblib.dump(ohe, 'ohe_class.pkl')
        else:
            ohe = joblib.load('ohe_class.pkl')
            ohe_array = ohe.transform(df[cat_cols])

        ohe_names = ohe.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(ohe_array, columns=ohe_names, index=df.index)
        df = pd.concat([df[all_numeric_cols], df_cat], axis=1)

    return df