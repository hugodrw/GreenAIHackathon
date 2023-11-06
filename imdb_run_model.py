import os
import pandas as pd
import joblib


# TEST_DATASET_URL = os.environ['TEST_DATASET_TEXT_URL']
TEST_DATASET_URL = 'sample_test.csv'


def evaluate():
    df = pd.read_csv(TEST_DATASET_URL)

    model = joblib.load('hugo_dfidf_svc_rbf_v1.bin')

    X_test = df['text']

    predictions = model.predict(X_test)
    
    df_predictions = pd.DataFrame({
        'id': df['id'],
        'label': predictions
    })

    df_predictions.to_csv('prediction.csv', index=False)
    
    print("success")


if __name__ == '__main__':
    evaluate()
