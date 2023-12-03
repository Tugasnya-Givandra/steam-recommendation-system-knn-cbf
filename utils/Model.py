import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import KNeighborsClassifier

class KnnCBF:
    def __init__(self, items, 
                item_col='app_id',
                score_col='is_recommended',
                nearest_k=2,
                metric="manhattan"):
        """
        Args:
            items:     (DataFrame) games dataframe contain tags attribute
            item_col:  (String) column name of items column
            score_col: (String) column name of interactions column
            k_nearest: (Integer) number of nearest interacted items for similarity
        """
        
        self.item_col = item_col
        self.score_col = score_col
        self.nearest_k = nearest_k
        self.metric = metric
        self.items = items.iloc[:,[0, *range(13, len(items.columns))]]

    def fit(self, X, y, test_data):
        classifier = KNeighborsClassifier(
                        n_neighbors=self.nearest_k, 
                        metric=self.metric
                        )
        classifier.fit(X, y)

        return classifier.kneighbors(test_data)
    
    
    def fit_predict(self, df_pred, k=10):
        select_row      = self.items['app_id'].isin(df_pred['app_id'])
        df_preferences  = self.items[select_row].merge(df_pred, on=['app_id']).iloc[:, 1:]
        df_test         = self.items[~select_row] # & filter_index]

        _output_preds = []
        _score_preds = []

        # Fitting using Features
        label = df_preferences['is_recommended']
        X = df_preferences.drop('is_recommended', axis=1)

        test = df_test.iloc[:, 1:]
        print(label)
        print(X)
        print(test)

        neighbor_distances, neighbor_indices = self.fit(X, label, test)

        rating = label.loc[neighbor_indices.flatten()] \
                        .values \
                        .reshape(neighbor_indices.shape)
        result = np.sum(rating * neighbor_distances, axis=1) / self.nearest_k

        top_tensor = torch.from_numpy(result).topk(k, largest=False)
        indices = top_tensor.indices.tolist()
        score = top_tensor.values.tolist()

        app_ids = self.items.iloc[indices, ]['app_id'].values
        print(app_ids)

        pred_result = pd.DataFrame({
            'app_id' : app_ids,
            'predicted_score' : score
        })
        
        return pred_result