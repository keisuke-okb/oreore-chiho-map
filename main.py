from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import random

import embed
import oreorechiho


def main():
    # ===== STEP1. キーワードの埋め込み, 主成分分析による次元削減 =====
    # マップサイズ（ピクセル）の設定
    map_width = 60
    map_height = 40

    # キーワードリストの読み込み
    li = pd.read_csv("list.csv", encoding="cp932", header=None)
    li = li.sort_values(0).reset_index(drop=True)
    keywords = li[0].values

    # 埋め込みモデルの初期化
    model = embed.EmbeddingModel(model_path="Z:\models\multilingual-e5-large")

    # キーワードを埋め込みベクトルに変換
    embeddings = model.embed(keywords)

    # 埋め込みベクトルを次元削減
    pca_result = model.pca(embeddings, map_width, map_height)

    # 次元削減結果をデータフレームに変換
    df_embeddings = model.convert_to_df(pca_result)

    # 結果をCSVファイルとして保存
    df_embeddings.to_csv("embeddings.csv", index=False)

    # ===== STEP2. クラスタ化、最小全域木による道の作成、起伏画像・地形画像を作成し地図を描画 =====
    # 埋め込みデータの読み込み
    df_embeddings = pd.read_csv("embeddings.csv")

    # 埋め込みデータの正規化と座標への変換
    embeddings = df_embeddings[["0", "1"]].values
    embeddings -= embeddings.min(axis=0)
    embeddings /= embeddings.max(axis=0)
    embeddings[:, 0] *= (map_width * 0.8 - 1)
    embeddings[:, 1] *= (map_height * 0.8 - 1)
    embeddings[:, 0] += map_width * 0.1
    embeddings[:, 1] += map_height * 0.1

    # 点をクラスタリングし、クラスタサイズを計算
    clusters, cluster_sizes = oreorechiho.cluster_points(embeddings, map_width, map_height)

    # クラスタ間を接続する道路データを生成
    roads = oreorechiho.generate_road_data(clusters)

    # キーワードを基にSHA256ハッシュからシードを生成
    keywords = df_embeddings["keywords"].values
    hash_input = "".join(keywords).encode('utf-8')
    unique_seed = hashlib.sha256(hash_input).hexdigest()

    # 点に基づいて地形の起伏画像を作成
    relief_img = oreorechiho.create_relief_image(map_width, map_height, embeddings)
    plt.imshow(relief_img)

    # シードを基にランダムな起伏画像を作成
    seed_relief_img = oreorechiho.create_seed_relief_image(map_width, map_height, int(unique_seed, 16))
    plt.imshow(seed_relief_img)

    # 砂地の地形画像を作成
    seed_sand_img = oreorechiho.create_seed_terrain_image(map_width, map_height, int(unique_seed, 16) + 1, 4, 5, 0.01, 0.1)
    plt.imshow(seed_sand_img)

    # 岩地の地形画像を作成
    seed_rock_img = oreorechiho.create_seed_terrain_image(map_width, map_height, int(unique_seed, 16) + 2, 5, 5, 0.01, 0.1)
    plt.imshow(seed_rock_img)

    # 起伏画像を統合し、地形種別を調整
    _relief_img_composite = relief_img + seed_relief_img
    _relief_img_composite[(_relief_img_composite <= 0) & (relief_img == 1)] = 1
    _relief_img_composite[(_relief_img_composite >= 1) & (seed_sand_img == 4)] = 4
    _relief_img_composite[(_relief_img_composite >= 1) & (seed_rock_img == 5)] = 5

    relief_img_composite = Image.fromarray(_relief_img_composite)

    # 地形種別に対応するカラーを設定
    color_map = {
        -1: (90, 0, 200),  # 深海
        0: (0, 0, 200),     # 海
        1: (0, 255, 0),     # 平地
        2: (0, 170, 0),   # 林、森
        3: (0, 100, 0),    # 山
        4: (250, 223, 102),    # 砂漠
        5: (80, 80, 80),    # 岩
        6: (217, 173, 130),    # 道
        7: (252, 76, 28), # 街
    }

    # 地形、道路、クラスタ情報を用いて最終的な地図画像を描画
    terrain_map, road_map, final_map = oreorechiho.draw_map(
        relief_img_composite,
        color_map,
        roads,
        [c['center'] for c in clusters],
        cluster_sizes,
        map_width,
        map_height,
        int(unique_seed, 16),
    )

    # 地形マップを拡大して保存
    terrain_map = terrain_map.resize((map_width * 20, map_height * 20), Image.Resampling.NEAREST)
    terrain_map.save('terrain_map.png', 'PNG')

    # 道路マップを拡大して保存
    road_map = road_map.resize((map_width * 20, map_height * 20), Image.Resampling.NEAREST)
    road_map.save('road_map.png', 'PNG')

    # 最終的なマップを拡大して保存
    final_map = final_map.resize((map_width * 20, map_height * 20), Image.Resampling.NEAREST)
    final_map.save('final_map.png', 'PNG')

    plt.imshow(terrain_map)

    # テクスチャ画像でマップを描画
    terrain_pixels = terrain_map.load()
    texture_map = Image.new("RGBA", terrain_map.size, (0, 0, 0, 255))
    texture_size = 40

    # 各地形種別に対応するテクスチャを適用
    for k in tqdm(color_map.keys()):
        try:
            item = Image.open(f"./images/{k}.png").resize((texture_size, texture_size))

            for _ in range(10000):
                x, y = random.randint(0, texture_map.size[0] - 1), random.randint(0, texture_map.size[1] - 1)
                if terrain_pixels[x, y][:3] == color_map[k]:
                    _img = Image.new("RGBA", texture_map.size, (255, 255, 255, 0))
                    _img.paste(item, (x - texture_size // 2, y - texture_size // 2))
                    texture_map = Image.alpha_composite(texture_map, _img)
        except:
            pass

    # テクスチャマップを保存
    texture_map.save('texture_map.png', 'PNG')
    plt.imshow(texture_map)


if __name__ == "__main__":
    main()