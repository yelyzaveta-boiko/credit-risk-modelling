# EDA based baseline feature set known at loan origination for the first PD model
# Human baseline, subject to revision after WOE/IV analysis

BASELINE_FEATURES = [
    "loan_amnt", # loan size
    "term", # 36 or 60 months
    "int_rate", # interest rate
    "grade", # Lending Club's own risk grade (A-G)
    "emp_length", # employment length
    "home_ownership", # rent / own / mortgage
    "annual_inc", # borrower income
    "verification_status",  # income verification status
    "purpose", # stated reason for the loan
    "dti", # debt to income ratio
    "fico_range_low", # FICO score at origination
    "open_acc", # number of open credit lines
    "pub_rec", # derogatory public records
    "revol_util", # revolving credit utilization
    "total_acc",  # total credit lines ever opened
]
