import new_lib as nl
import pandas as pd
import os


def get_telco_data(get_db_url):
    
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