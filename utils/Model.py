import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

from utils.utils import get_recommendation_dataset

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
        self.app_ids = items[item_col]
        self.items = items.iloc[:, 13:]

        svd = TruncatedSVD(n_components=40, random_state=1)
        self.items_decomposed = pd.DataFrame(svd.fit_transform(self.items))
        self.items_decomposed.columns = self.items_decomposed.columns.astype(str)
        self.items_decomposed[item_col] = self.app_ids
        print(self.items_decomposed)

    def fit(self, X, y, test_data):
        classifier = KNeighborsClassifier(
                        n_neighbors=self.nearest_k, 
                        metric=self.metric
                        )
        classifier.fit(X, y)

        return classifier.kneighbors(test_data)
    
    
    def fit_predict(self, df_pred, k=10):
        select_row      = self.app_ids.isin(df_pred['app_id'])
        df_preferences  = self.items_decomposed[select_row].merge(df_pred, on=['app_id'])
        df_test         = self.items_decomposed[~select_row] # & filter_index]

        # Fitting using Features

        label = df_preferences['is_recommended']
        X = df_preferences.iloc[:, :-2]
        print("+++++++++++++++++++++++")
        print(label)
        print(X)
        # df_preferences = df_preferences.iloc[:, :-1]

        test = df_test.iloc[:, :-1]

        neighbor_distances, neighbor_indices = self.fit(X, label, test)

        rating = label.loc[neighbor_indices.flatten()] \
                        .values \
                        .reshape(neighbor_indices.shape)
        result = np.sum(rating * neighbor_distances, axis=1)

        top_tensor = torch.from_numpy(result).topk(k, largest=False)
        indices = top_tensor.indices.tolist()
        score = top_tensor.values.tolist()

        app_ids = self.items_decomposed.iloc[indices]['app_id'].values
        print(app_ids)
        print(score)

        pred_result = pd.DataFrame({
            'app_id' : app_ids,
            'predicted_score' : score
        })
        
        return pred_result
    
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

class KnnCBFUser:
    def __init__(self, 
                 sample_size=75000,
                 k=30,
                 N=10
                 ) -> None:
        
        self.sample_size = sample_size
        self.k = k
        self.N = N


    def fit_predict(self, user_row):
        user_idx = 99999999999
        user_row['user_id'] = user_idx
        df = get_recommendation_dataset()
        df_user = pd.read_csv('datasets/archive/users.csv')

        # print(df.dtypes)
        # Randomly select a subset of users
        # random_users = df['user_id'].sample(n=int(self.sample_size)) #, random_state=42)
        print(df_user.columns)
        drop_list = df_user[(df_user['products'] > 5) & (df_user['reviews'] > 2)]
        print(len(drop_list))


        # Filter the DataFrame to include only the selected users
        df_subset = df[~df['user_id'].isin(drop_list) ]
        print(len(df_subset))
        print()
        df_subset = pd.concat([df_subset, user_row], axis=0).reset_index(drop=True)
        # print(len(random_users))
        print(len(df_subset))

        # Create a user-item matrix from the subset
        user_u = list(sorted(df_subset['user_id'].unique()))
        user_u_ori = list(sorted(df['user_id'].unique()))
        item_u = list(sorted(df['app_id'].unique()))

        print("User Percentage", len(user_u) / len(user_u_ori) * 100, '%')

        row = df_subset['user_id'].astype('category').cat.codes
        col = df_subset['app_id'].astype('category').cat.codes

        data = df_subset['is_recommended'].astype(int).tolist()

        user_item_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))

        user_similarity = cosine_similarity(user_item_matrix, dense_output=False)

        def get_top_k_neighbors(user_similarity_matrix, target_user_idx, k):
            # Extract the row of cosine similarities for the target user
            target_similarities = user_similarity_matrix[target_user_idx, :]

            top_k_indices = np.argsort(target_similarities.data)[-k:][::-1]
            top_k_similarities = target_similarities.data[top_k_indices]

            return top_k_indices, top_k_similarities

        print('choices', user_item_matrix[len(user_u)-1, :])

        top_k_neighbors, top_k_similarities = get_top_k_neighbors(
                                                user_similarity, 
                                                len(item_u)-1, 
                                                self.k)

        def predict_ratings(user_item_matrix, top_k_neighbors, top_k_similarities):
            # Get the ratings of the top-K neighbors
            neighbor_ratings = user_item_matrix[top_k_neighbors, :]
            # print(neighbor_ratings)
            
            # Weighted sum of neighbor ratings based on similarity
            non_weighted_sum = neighbor_ratings.sum(axis=0)
            # print(type(neighbor_ratings))
            # print(non_weighted_sum)
            # print(weighted_sum.shape)

            # Sum of absolute similarities for normalization
            abs_sim_sum = np.sum(np.abs(top_k_similarities))

            # Predicted ratings for the target user
            # predicted_ratings = non_weighted_sum / abs_sim_sum

            return non_weighted_sum

        predicted_ratings = predict_ratings(
                                user_item_matrix, 
                                top_k_neighbors, 
                                top_k_similarities)
        
        pr = predicted_ratings.tolist()[0]
        pr.sort(reverse=True)
        
        top_recommendations = np.argsort(predicted_ratings.data)[::-1][:self.N]
        print("Top Recommendations:", top_recommendations)
        print("weight             :", pr[:self.N])

        rec_indices = top_recommendations.tolist()
        rec_app_ids = itemgetter(*rec_indices[0])(item_u)

        return rec_app_ids[:self.N]

