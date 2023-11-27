import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import KNeighborsClassifier

class KnnCBF:
    def __init__(self, items, 
                user_col='user_id', 
                item_col='app_id',
                score_col='is_recommended',
                nearest_k=3,
                metric="cosine"):
        """
        Args:
            items:     (DataFrame) games dataframe contain tags attribute
            user_col:  (String) column name of users column
            item_col:  (String) column name of items column
            score_col: (String) column name of interactions column
            k_nearest: (Integer) number of nearest interacted items for similarity
        """
        
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.nearest_k = nearest_k
        self.metric = metric

        self.item_id_col = item_col + "_index"

        self.item_lookup = self.generate_label(items, self.item_col)

        self.item_map = {}
        for item, item_index in self.item_lookup.values:
            self.item_map[item_index] = item

        # Creating similarity items
        items = items.merge(self.item_lookup, on=[self.item_col], sort=False)
        items = items.drop(items.columns[:2], axis=1)

        # Reindexing items dataframe
        cols = list(items.columns)
        items = items[cols[-1:] + cols[:-1]]
        self.items = items
    

    def generate_label(self, df, col):
        dist_labels = df[[col]].drop_duplicates()
        dist_labels[col + "_index"] = dist_labels[col] \
                                        .astype("category") \
                                        .cat.codes

        return dist_labels

    def fit(self, feature_vector, y, test_data):
        classifier = KNeighborsClassifier(
                        n_neighbors=self.nearest_k, 
                        metric=self.metric
                        )
        classifier.fit(feature_vector, y)

        return classifier.kneighbors(test_data)
    
    def fit_predict(self, df_pred, k=10):
        df_active = df_pred.merge(self.item_lookup, on=[self.item_col], sort=False)
        df_pred = df_pred[[self.user_col]].drop_duplicates()

        df_active = df_active[[self.item_id_col, self.score_col]]       
        # ----------------------------------------------------------------------  
        
        active_items = self.items.merge(df_active, on=[self.item_id_col], sort=False)
        inactive_items = self.items[~self.items['app_id_index'].isin(df_active['app_id_index'])]

        _output_preds = []
        _score_preds = []

        # Fitting using Features
        X = active_items.iloc[:, 1:-1]
        print(X)
        y = active_items.iloc[:, -1]
        print(y)
        test = inactive_items.iloc[:, 1:]
        print(test)

        output = self.fit(X, y, test)

        rating = y.loc[output[1].flatten()] \
                  .values \
                  .reshape(output[1].shape)
        result = np.sum(rating * output[0], axis=1) / self.nearest_k

        self.preds_tensor_ = result

        top_tensor = torch.from_numpy(result).topk(k)
        indices = top_tensor.indices.tolist()
        score = top_tensor.values

        _output_preds.append( [self.item_map[_id] for _id in indices] )

        _score_preds.append( score.tolist() )

        df_pred['predicted_items'] = _output_preds
        df_pred['predicted_score'] = _score_preds

        escaped_id = [
            ele for i_list in df_pred['predicted_items'].values for ele in i_list
        ]

        escaped_score = [
            score for s_list in df_pred['predicted_score'].values for score in s_list
        ]

        pred_result = pd.DataFrame({
            'app_id' : escaped_id,
            'predicted_score' : escaped_score
        })
        
        return pred_result