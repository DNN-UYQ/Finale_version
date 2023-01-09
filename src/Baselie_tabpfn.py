from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import openml
import numpy as np

dataset_id= np.load("id_test_datasets.npy", allow_pickle=True).tolist()

for dataset in dataset_id:
    print(dataset)
    try:
        test_dataset = openml.datasets.get_dataset(dataset_id=dataset)
        X, y, categorical_indicator, attribute_names = test_dataset.get_data(dataset_format="array",
                                                                                 target=test_dataset.default_target_attribute)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42, stratify=y)
        classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        print(X_train.shape)
        classifier.fit(X_train, y_train)
        y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
        print('Accuracy', accuracy_score(y_test, y_eval))
    except:
        print("next plz")
