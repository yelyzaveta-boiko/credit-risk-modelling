# EDA based baseline feature set known at loan origination for the first PD model
# Human baseline, subject to revision after WOE/IV analysis

NUMERIC_FEATURES = [
    "loan_amnt", # loan size
    "int_rate", # interest rate
    "annual_inc", # borrower income
    "dti", # debt to income ratio
    "fico_range_low", # FICO score at origination
    "open_acc", # number of open credit lines
    "pub_rec", # derogatory public records
    "revol_util", # revolving credit utilization
    "total_acc",  # total credit lines ever opened
]

CATEGORICAL_FEATURES = [
    "term", # 36 or 60 months
    "grade", # Lending Club's own risk grade (A-G)
    "emp_length", # employment length
    "home_ownership", # rent / own / mortgage
    "verification_status",  # income verification status
    "purpose", # stated reason for the loan
]

BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
