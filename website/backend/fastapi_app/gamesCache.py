import pandas as pd
import numpy as np
from scipy.linalg import svd


class GamesCache:
    def __init__(self, game_data: pd.DataFrame):
        self.game_data = game_data
        self.tag_names = None
        self.IDF = None
        self.mu = None
        self.tf_idf_centered = None
        self.U = None
        self.S = None
        self.V = None
        self.latent_data = None
        self.calc_fields()

    def calc_fields(self):
        """Calculate fields using TF-IDF and SVD."""
        last_col_before_data = self.game_data.columns.get_loc("IsAGame")
        data_start_col = last_col_before_data + 1
        self.tag_names = self.game_data.columns[data_start_col:].tolist()
        tag_data = self.game_data.iloc[:, data_start_col:].to_numpy()
        tf_idf, self.IDF = GamesCache.get_tf_idf(tag_data)
        self.mu = np.mean(tf_idf, axis=0)
        self.tf_idf_centered = tf_idf - self.mu
        self.U, self.S, self.V = svd(self.tf_idf_centered, full_matrices=False)
        latent_space_coordinates = self.U @ np.diag(self.S)
        latent_names = [f"dim {i + 1}" for i in range(latent_space_coordinates.shape[1])]
        latent_space_coordinates[np.abs(latent_space_coordinates) < 1e-10] = 0
        self.latent_data = pd.DataFrame(latent_space_coordinates, columns=latent_names)

    @staticmethod
    def get_tf_idf(mat):
        """Compute TF-IDF representation."""
        total_tags = np.maximum(np.sum(mat, axis=1, keepdims=True), 1)
        TF = mat / total_tags
        DF = np.sum(mat > 0, axis=0)
        num_documents = mat.shape[0]
        IDF = np.log(num_documents / DF)
        tfidf = TF * IDF
        tfidf[np.isnan(tfidf) | np.isinf(tfidf)] = 0
        return tfidf, IDF
