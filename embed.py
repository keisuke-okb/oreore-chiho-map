import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import japanize_matplotlib


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    トークンの隠れ状態を平均プールすることで文全体の埋め込みを計算します。

    Args:
        last_hidden_states (Tensor): 隠れ層の最終状態。
        attention_mask (Tensor): アテンションマスク。

    Returns:
        Tensor: 平均プールされた文埋め込み。
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel:
    """
    文埋め込みモデルを定義するクラス。
    モデルの初期化、キーワードの埋め込み生成、次元削減を行います。
    """

    def __init__(self, model_path):
        """
        モデルとトークナイザーを初期化します。

        Args:
            model_path (str): 事前学習済みモデルのパス。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to("cuda:0")
        self.keywords = None

    def embed(self, keywords: list) -> np.array:
        """
        キーワードリストを埋め込みベクトルに変換します。

        Args:
            keywords (list): 埋め込み対象のキーワード。

        Returns:
            np.array: 埋め込みベクトル。
        """
        self.keywords = keywords
        input_texts = [f'query: {keyword}' for keyword in keywords]
        batch_dict = self.tokenizer(
            input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt'
        ).to("cuda:0")
        outputs = self.model(**batch_dict)

        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
        return embeddings

    def pca(self, embeddings: np.array, map_width, map_height) -> np.array:
        """
        次元削減(PCA)を用いて埋め込みベクトルを2次元に変換します。

        Args:
            embeddings (np.array): 高次元埋め込みベクトル。

        Returns:
            np.array: 2次元に変換された埋め込みベクトル。
        """
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)

        pca_result -= pca_result.min(axis=0)
        pca_result /= pca_result.max(axis=0)
        pca_result[:, 0] *= (map_width * 0.8 - 1)
        pca_result[:, 1] *= (map_height * 0.8 - 1)
        pca_result[:, 0] += map_width * 0.1
        pca_result[:, 1] += map_height * 0.1

        # 次元削減した結果をプロットします。
        plt.figure(figsize=(map_width // 10, map_height // 10))
        for i, label in enumerate(self.keywords):
            plt.scatter(pca_result[i, 0], pca_result[i, 1])
            plt.text(pca_result[i, 0], pca_result[i, 1], label)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA on 1024D Vectors')
        plt.xlim(0, map_width)
        plt.ylim(0, map_height)
        plt.gca().invert_yaxis()
        plt.show()
        return pca_result

    def convert_to_df(self, pca_result: np.array) -> pd.DataFrame:
        """
        次元削減結果をデータフレーム形式に変換します。

        Args:
            pca_result (np.array): 次元削減結果。

        Returns:
            pd.DataFrame: キーワードと座標を含むデータフレーム。
        """
        df_embeddings = pd.concat([pd.DataFrame(self.keywords), pd.DataFrame(pca_result)], axis=1)
        df_embeddings.columns = ["keywords", "0", "1"]
        return df_embeddings


if __name__ == "__main__":
    # マップサイズ（ピクセル）の設定
    map_width = 60
    map_height = 40

    # キーワードリストの読み込み
    li = pd.read_csv("list.csv", encoding="cp932", header=None)
    li = li.sort_values(0).reset_index(drop=True)
    keywords = li[0].values

    # 埋め込みモデルの初期化
    model = EmbeddingModel(model_path="Z:\models\multilingual-e5-large")

    # キーワードを埋め込みベクトルに変換
    embeddings = model.embed(keywords)

    # 埋め込みベクトルを次元削減
    pca_result = model.pca(embeddings, map_width, map_height)

    # 次元削減結果をデータフレームに変換
    df_embeddings = model.convert_to_df(pca_result)

    # 結果をCSVファイルとして保存
    df_embeddings.to_csv("embeddings.csv", index=False)
