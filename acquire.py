import new_lib as nl
import pandas as pd
import os


def get_telco_data(get_db_url):
    ''' Acquiring the raw telco data
    
    '''
    if os.path.isfile('telco.csv'):
        
        return pd.read_csv('telco.csv')
    
    else:
        url = get_db_url('telco_churn')
        query = '''SELECT * FROM customers
                    JOIN internet_service_types USING(internet_service_type_id)
                    JOIN contract_types USING(contract_type_id)
                    JOIN payment_types USING(payment_type_id)
                    '''
        df = pd.read_sql(query, url)
        df.to_csv('telco.csv')
        return df

def data(df):
    ''' Curating the telco data to the desired shape and
        list of parameters
    
    '''
    df = get_telco_data(nl.get_db_url)
    df = df.iloc[:, 1:]
    df = df.drop(['gender', 'senior_citizen', 'partner', 'phone_service', 'tech_support', 'streaming_tv', 
                    'streaming_movies', 'paperless_billing', 'internet_service_type', 'online_security', 'online_backup', 
                    'device_protection', 'internet_service_type_id', 'customer_id', 'multiple_lines'], axis =1)
    df = df.dropna()
    df.total_charges = df.total_charges.replace(' ', 0).astype(float)
    
    return df