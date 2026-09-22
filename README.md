# pokemon_kunn — ポケモンバトル AI

ロードマップ「ポケモンバトルAI 実装ロードマップ（PPO→Rainbow→PPO×MCTS→LLM）」の実装リポジトリです。
現在は **Phase 0（対戦環境＋評価）** と **Phase 1（PPO による強化学習）** まで実装されています。

## 概要

- **対戦形式:** Pokémon Showdown の `[Gen 9 Champions] Random Battle`（フォーマット ID `gen9championsrandombattle`、
  セットは [data/random-battles/champions](https://github.com/smogon/pokemon-showdown/tree/master/data/random-battles/champions)）。
  - シングル 6vs6、チームはランダム生成、Lv 44〜60
  - メガシンカあり／テラスタルなし、技の PP は最大 20（Champions 独自ルール）
- **行動空間:** poke-env の 26 アクション配置をそのまま使い、使えないものはマスクで禁止します。

  | アクション番号 | 意味 |
  |---|---|
  | 0〜5 | 手持ち 0〜5 番に交代 |
  | 6〜9 | 場のポケモンの技 1〜4 |
  | 10〜13 | メガシンカ＋技 1〜4 |
  | 14〜25 | Z技・ダイマックス・テラスタル（Champions では常にマスク＝使用不可） |

- **全体の構成:**

```
┌──────────────────────── ホスト (Mac) ─────────────────────────┐
│  uv run python -m pokeai.ppo                                   │
│   ├─ 学習ループ (PPO, PyTorch)                                   │
│   └─ 並列環境 × num_envs（サブプロセス）                          │
│        1 環境 = 学習エージェント + 対戦相手 の 2 アカウントで接続    │
│             │ WebSocket (ws://localhost:8000/showdown/websocket) │
└─────────────┼──────────────────────────────────────────────────┘
              ▼
┌──────── Docker: pokemon-showdown コンテナ ────────┐
│  Pokémon Showdown サーバ（対戦シミュレータ本体）      │
│   /app/config    ← data/config                   │
│   /app/databases ← data/databases                │
│   /app/logs/*    ← data/logs/*                   │
└──────────────────────────────────────────────────┘
```

ダメージ計算やターン処理などバトルのルールはすべて Showdown サーバが担当し、
Python 側（[poke-env](https://github.com/hsahovic/poke-env)）はサーバから届く対戦状況を受け取って、
選んだ行動を送り返すだけです。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `docker-compose.yml`, `docker/server/` | ローカルの Pokémon Showdown サーバ（`--no-security` 付き: 認証・連投制限なし） |
| `docker/client/` | poke-env の RandomPlayer 同士を 1 戦させる疎通確認用クライアント |
| `src/pokeai/` | 学習・評価の Python コード本体（[6 章](#6-srcpokeai--コードの概要)） |
| `configs/` | 学習設定 YAML（[1 章](#1-configs--学習設定ファイル)） |
| `tests/test_encoding.py` | 観測エンコーダとアクションマスクの単体テスト（サーバ不要） |
| `data/` | Showdown サーバの設定・DB・ログ（コンテナにマウント。`.gitignore` 対象） |
| `runs/` | 学習結果の出力先（`.gitignore` 対象） |

---

## 1. `configs/` — 学習設定ファイル

### 目的

PPO 学習の「何ステップ学習するか」「並列数」「ハイパーパラメータ」「報酬の形」「モデルの大きさ」「どの相手と戦わせるか（カリキュラム）」を
YAML 1 枚にまとめたものです。`python -m pokeai.ppo --config <ファイル>` で渡すと
`src/pokeai/ppo.py` の `TrainConfig.load()` が読み込み、書かれていない項目は `TrainConfig` / `EnvConfig` / `ModelConfig` の既定値が使われます。
実際に使われた設定（既定値込み）は `runs/<run>/config.yaml` に保存されるので、後から再現できます。

| ファイル | 用途 |
|---|---|
| `ppo_smoke.yaml` | パイプライン全体の動作確認用（数分で終わる）。カリキュラム切替・self-play スナップショット・チェックポイント・TensorBoard が一通り動くかを見るだけで、強くはならない |
| `ppo_default.yaml` | Phase 1 の本番学習（500 万ステップ）。目標は SimpleHeuristicsPlayer に 500 戦で勝率 60% 以上 |
| `ppo_noshaping.yaml` | アブレーション実験: 報酬を勝敗（±1）のみにして、途中報酬（shaping）の効果を比べる。それ以外は default と同じ |

### パラメータ一覧

「既定値」は YAML に書かなかったときにコード側で使われる値です。

#### 実行全体

| キー | 既定値 | 説明 |
|---|---|---|
| `run_name` | `ppo` | 出力ディレクトリ名の接頭辞。`runs/<run_name>-<日時>/` になる |
| `seed` | `1` | 乱数シード（Python / NumPy / PyTorch / 環境リセット） |
| `device` | `auto` | `auto` なら CUDA があれば GPU、なければ CPU（モデルが小さいので Mac では MPS より CPU の方が速く、自動では MPS を選ばない）。`cpu` / `cuda` / `mps` を直接指定も可 |
| `total_timesteps` | `1_000_000` | 学習する総ステップ数（1 ステップ = 1 環境での 1 回の行動選択）。CLI の `--total-timesteps` で上書き可 |
| `num_envs` | `8` | 並列で走らせる対戦環境の数。各環境は別プロセスで Showdown に 2 接続（自分と相手）する。CLI の `--num-envs` で上書き可 |
| `num_steps` | `256` | 1 回の更新の前に各環境で集めるステップ数。**バッチサイズ = `num_envs × num_steps`**（default 設定なら 16×256 = 4096） |

#### PPO ハイパーパラメータ

| キー | 既定値 | 説明 |
|---|---|---|
| `learning_rate` | `2.5e-4` | Adam の学習率 |
| `anneal_lr` | `true` | 学習率を学習の進行に合わせて 0 まで線形に下げる |
| `gamma` | `0.99` | 割引率。将来の報酬をどれだけ重視するか |
| `gae_lambda` | `0.95` | GAE（Generalized Advantage Estimation）の λ。アドバンテージ推定のバイアスと分散のバランス |
| `num_minibatches` | `4` | 1 バッチをいくつのミニバッチに分けて勾配更新するか |
| `update_epochs` | `4` | 集めた 1 バッチを何周して学習するか |
| `norm_adv` | `true` | ミニバッチ内でアドバンテージを正規化する（YAML では未指定＝既定値） |
| `clip_coef` | `0.2` | PPO のクリップ幅 ε。方策が 1 回の更新で変わりすぎないように制限する |
| `clip_vloss` | `true` | 価値関数の損失にもクリップを掛ける（YAML では未指定） |
| `ent_coef` | `0.01` | エントロピー項の係数。大きいほどランダムに探索する |
| `vf_coef` | `0.5` | 価値関数損失の係数 |
| `max_grad_norm` | `0.5` | 勾配クリッピングの上限ノルム |
| `target_kl` | なし | 更新中に方策の変化（近似 KL）がこの値を超えたら、そのバッチの残りのエポックを打ち切る（default では `0.05`） |

#### 保存・self-play

| キー | 既定値 | 説明 |
|---|---|---|
| `checkpoint_every_updates` | `20` | 何回の更新ごとに `checkpoints/step_XXXXXXXXX.pt` を保存するか（`latest.pt` は毎更新上書き） |
| `snapshot_every_updates` | `20` | 何回の更新ごとに現在のモデルを凍結して self-play プール `pool/` に追加するか（default では 25 更新 ≒ 10 万ステップごと） |
| `pool_size` | `20` | self-play プールに残す過去モデルの最大数（古いものから外れる） |

#### `env:` — 対戦環境

| キー | 既定値 | 説明 |
|---|---|---|
| `battle_format` | `gen9championsrandombattle` | Showdown のフォーマット ID |
| `reward.victory` | `1.0` | 勝ちで +victory、負けで −victory（最終報酬） |
| `reward.fainted` | `0.0` | ひんしに対する途中報酬（相手をひんしにすると +、自分がひんしになると −） |
| `reward.hp` | `0.0` | 両陣営の HP 割合に対する途中報酬 |
| `reward.status` | `0.0` | 状態異常に対する途中報酬（どの YAML でも未使用） |

途中報酬は poke-env の `reward_computing_helper` で計算され、「前ステップからの差分」が毎ステップ報酬として与えられます。
default では `{victory: 1.0, fainted: 0.05, hp: 0.05}`、noshaping では fainted/hp を 0 にしています。

#### `model:` — ネットワークの大きさ

| キー | 既定値 | 説明 |
|---|---|---|
| `d_model` | `128` | Transformer の隠れ次元（トークンのベクトル長） |
| `n_layers` | `2` | Transformer エンコーダ層の数 |
| `n_heads` | `4` | アテンションのヘッド数 |
| `species_dim` / `move_dim` / `hashed_dim` | `64` / `32` / `16` | ポケモン種族・技・道具/特性の埋め込み次元（YAML では未指定） |

既定値でのパラメータ数は約 68 万です。

#### `curriculum:` — 対戦相手のカリキュラム

`{until_step, mix}` のリストで、学習の進み具合に応じて対戦相手の出現比率を切り替えます。
`until_step` を省略したステージは学習終了まで続きます。対戦相手は 1 試合ごとに `mix` の重みで抽選されます。

| `mix` のキー | 相手 |
|---|---|
| `random` | 合法手からランダムに選ぶ（poke-env `RandomPlayer`） |
| `max_power` | 威力が最大の技を選ぶ（`MaxBasePowerPlayer`） |
| `heuristic` | タイプ相性や交代などを考慮するルールベース（`SimpleHeuristicsPlayer`）。Phase 1 の目標となる相手 |
| `latest` | 自分自身の最新スナップショット（self-play） |
| `pool` | self-play プールからランダムに選んだ過去の自分 |
| `ckpt:<path>` | 任意のチェックポイントファイル |

`ppo_default.yaml` のカリキュラム:

1. 〜30 万ステップ: `random` 50% / `max_power` 50% — 弱い相手で基本を覚える
2. 〜150 万ステップ: `max_power` 30% / `heuristic` 70% — 本命の相手に慣れる
3. それ以降: `heuristic` 20% / `latest` 40% / `pool` 40% — self-play。heuristic を少し残して目標に対する勝率を測り続ける

---

## 2. `data/config/` — Showdown サーバの設定

`docker-compose.yml` でコンテナの `/app/config` にマウントされる、**Pokémon Showdown サーバ自身の設定ディレクトリ**です。
Python の学習コードは直接読みません。

- 中身は Showdown 公式リポジトリの `config/` とほぼ同じです。初回起動時に `docker/server/entrypoint.sh` が
  イメージ内の既定ファイルを空のディレクトリへコピーし、`config.js` が無ければ `config-example.js` から作ります。
  ホスト側に置くことで、コンテナを作り直しても設定が残り、再ビルドせずに編集できます。
- 主なファイル:

| ファイル | 役割 |
|---|---|
| `config.js` | サーバ設定本体（ポート 8000 など）。`config-example.js` から 2 点を変更済み: `loginserver = null`（公式ログインサーバに問い合わせない）、`noguestsecurity = true`（パスワードなしで好きな名前のゲストとして入れる）。これで poke-env のボットが認証なしで接続できる |
| `config-example.js` | 公式の設定テンプレート（比較・復元用） |
| `formats.ts` | 対戦フォーマットの一覧。学習で使う `[Gen 9 Champions] Random Battle`（`mod: 'champions'`, `team: 'random'`）もここで定義されている |
| `CUSTOM-RULES.md` | カスタムルールの書き方（公式ドキュメント） |
| `chatrooms.json`, `avatars.json`, `suspects.json`, `chat-plugins/`, `ladders/`, `avatars/` | チャットルーム・アバター・ラダーなど、サーバ運営用の状態ファイル。学習には関係しない |
| `hosts.csv`, `proxies.csv` | IP/ホストの分類リスト（荒らし対策用）。ローカル利用では実質未使用 |

なお、連続対戦に必須の「連投制限の解除（nothrottle）」は `config.js` ではなく
`docker-compose.yml` の起動コマンド `start --no-security` で有効にしています
（これが無いと「前回の挑戦から 10 秒以内の挑戦」が拒否され、環境のリセットが止まります）。
`--no-security` は認証やチェックを全部外すので、このサーバは外部に公開しないでください。

## 3. `data/databases/` — Showdown サーバの SQLite データベース

コンテナの `/app/databases` にマウントされる、**Showdown サーバが内部で使う SQLite DB の置き場所**です。

| パス | 役割 |
|---|---|
| `offline-pms.db` | サーバが起動時に作る DB（オフラインのユーザー宛て PM と PM 設定）。学習では使われない |
| `schemas/*.sql` | サーバが DB を作るときのテーブル定義（フレンド、PM、モデレーションログ、保存バトル、チームなど） |
| `migrations/**` | DB スキーマのバージョンアップ用 SQL |

学習結果や対戦記録はここには保存されません（学習の出力はすべて `runs/` 側）。
ホストに置いているのは、コンテナを作り直しても DB が消えないようにするためです。

## 4. `data/logs/` — Showdown サーバのログ

コンテナの `/app/logs/chat`, `/app/logs/modlog`, `/app/logs/tickets` にマウントされる、**Showdown サーバのログ出力先**です。

| パス | 内容 |
|---|---|
| `chat/` | チャットログ（`config.js` の `logchat = false` なので通常は空） |
| `modlog/` | モデレーション操作のログ |
| `tickets/` | ヘルプチケットのログ |

いずれもサーバ運営用で、ローカルでボット同士が対戦するだけの用途ではほぼ空のままです。
`logs/repl`（デバッグ用 UNIX ソケットの置き場）は、macOS の Docker Desktop がソケットファイルを扱えないため意図的にマウントしていません。
**学習のログ（損失・勝率など）はここではなく `runs/<run>/tb/`（TensorBoard）と標準出力に出ます。**

---

## 5. 学習させるときに何を実行するか

### 手順

```sh
# 1. Showdown サーバを起動（localhost:8000、healthcheck 付き）
docker compose up -d showdown

# 2. Python の依存関係をインストール
uv sync

# 3. （任意）単体テスト（サーバ不要）
uv run pytest

# 4. まず smoke 設定でパイプライン全体が動くか確認（数分）
uv run python -m pokeai.ppo --config configs/ppo_smoke.yaml

# 5. 本番学習（Mac ではスリープで止まらないよう caffeinate を付ける）
caffeinate -i uv run python -m pokeai.ppo --config configs/ppo_default.yaml

# 6. 学習の様子を見る
uv run tensorboard --logdir runs

# 途中から再開（モデル＋オプティマイザ＋ステップ数を復元）
uv run python -m pokeai.ppo --config configs/ppo_default.yaml --resume runs/<run>/checkpoints/latest.pt
```

サーバのアドレスは環境変数 `SHOWDOWN_HOST` / `SHOWDOWN_PORT` で変更できます（既定 `localhost:8000`）。
`docker compose up client` を実行すると RandomPlayer 同士で 1 戦し、サーバとの疎通を確認できます。

### 出力（`runs/<run_name>-<日時>/`）

| パス | 内容 |
|---|---|
| `config.yaml` | 実際に使われた設定（既定値込み） |
| `tb/` | TensorBoard ログ。`win_rate/<相手>` は相手ごとの直近 200 試合の学習中勝率。ほかに `losses/*`, `charts/SPS`（1 秒あたりステップ数）, `charts/battle_turns` など |
| `checkpoints/latest.pt` | 最新モデル（毎更新上書き。`--resume` に使える） |
| `checkpoints/step_*.pt` | 定期保存のチェックポイント |
| `pool/step_*.pt` | self-play 用に凍結した過去モデル |

### 評価

```sh
# random / max_power / heuristic とそれぞれ 500 戦
# （勝率＋95% Wilson 信頼区間、平均ターン数、1 手あたりの推論時間）
uv run python -m pokeai.evaluate runs/<run>/checkpoints/latest.pt -n 500

# 総当たり戦＋Elo レーティング（random を 1000 に固定）
uv run python -m pokeai.evaluate --round-robin runs/a/checkpoints/latest.pt runs/b/checkpoints/latest.pt -n 200
```

結果はチェックポイントと同じフォルダに `eval_bench_*.json` / `eval_rr_*.json` として保存されます。
Phase 1 の目標は **heuristic に 500 戦で勝率 60% 以上**です。

### どうやって実現しているか（学習の流れ）

1. **並列環境の起動**（`ppo.py` → `env.make_env`）
   `gymnasium.vector.AsyncVectorEnv` で `num_envs` 個のサブプロセスを立ち上げます。
   各プロセスでは `PokemonEnv`（poke-env の `SinglesEnv` を継承）が Showdown に「学習エージェント用」と「対戦相手用」の 2 アカウントで接続し、
   その 2 アカウント同士で対戦します。相手側は、`OpponentMixEnv` が試合ごとにカリキュラムの `mix` から選んだプレイヤー（ルールベース or 過去の自分）が操作します。
2. **観測の作成**（`encoding.encode_battle`）
   毎ターン、poke-env が保持している対戦状況（両チーム・判明している相手の情報・天候/フィールド・技の情報など）を 2373 次元の数値ベクトルに変換し、
   同時に「今選べる行動」を表す 26 次元のマスクも作ります。
3. **行動選択とデータ収集（ロールアウト）**
   `ActorCritic` モデルがマスク付きで行動の確率分布と状態価値を出し、行動をサンプリングして全環境に送ります。
   これを `num_steps` 回繰り返し、観測・行動・報酬・価値などをバッファに貯めます。試合が終わった環境は自動でリセットされ、次の相手が抽選されます。
4. **アドバンテージ計算**
   貯めたデータから GAE（γ=`gamma`, λ=`gae_lambda`）で「その行動が期待よりどれだけ良かったか」を計算します。
5. **PPO 更新**
   バッチをシャッフルしてミニバッチに分け、`update_epochs` 周、「クリップ付き方策損失 ＋ 価値損失 − エントロピーボーナス」を最小化します。
   近似 KL が `target_kl` を超えたら早めに打ち切ります。
6. **ログ・保存・self-play**
   毎更新、TensorBoard とコンソールに損失・勝率を出力し、`latest.pt` を保存します。
   `snapshot_every_updates` ごとに現在のモデルを `pool/` に保存し、全環境に新しいプールを通知します（`latest` / `pool` の相手はこのプールから読み込まれる）。
   ステップ数がカリキュラムの境目を越えたら、全環境の対戦相手の比率を切り替えます。
7. これを `total_timesteps ÷ (num_envs × num_steps)` 回（default なら 1220 回）繰り返します。

参考: M2 Mac・16 環境で約 430 ステップ/秒。ボトルネックは Showdown ではなく Python 側です。

---

## 6. `src/pokeai/` — コードの概要

| ファイル | 役割 |
|---|---|
| `__init__.py` | パッケージ宣言のみ |
| `server.py` | Showdown への接続設定。`SHOWDOWN_HOST` / `SHOWDOWN_PORT` から WebSocket URL を作る `server_configuration()` と、名前が衝突しないようランダム接尾辞を付けたアカウントを作る `account()` |
| `encoding.py` | 対戦状況 → 観測ベクトルの変換 |
| `env.py` | Gymnasium 環境（報酬計算、対戦相手の切り替え） |
| `model.py` | 方策・価値ネットワーク（Actor-Critic）とチェックポイントの保存/読み込み |
| `opponents.py` | 対戦相手（ルールベース・学習済みモデル・self-play プール）の生成 |
| `ppo.py` | PPO 学習のメインスクリプト（`python -m pokeai.ppo`） |
| `evaluate.py` | ベンチマーク評価と Elo 計算（`python -m pokeai.evaluate`） |

### `encoding.py` — 観測エンコーディング

バトルの状態を固定長（2373 次元）の float32 ベクトルにします。中身は 4 つのブロックに分かれています。

```
[GLOBAL 81] [ポケモン数値 48 × 12体] [ポケモンID 7 × 12体] [技数値 34 × 12体 × 4技]
```

- **スロット:** 0〜5 が自分のチーム（交代アクション 0〜5 と同じ順番）、6〜11 がこれまでに判明した相手のポケモン。
  技も `pokemon.moves` の順（技アクション 6〜9 と同じ順）に並べ、モデルが「どの行動がどのトークンに対応するか」を直接参照できるようにしています。
- **GLOBAL:** 天候、フィールド、両陣営の場の状態（まきびし等は段階数）、ターン数、メガシンカ可能/使用済み（自分・相手）、残りポケモン数、強制交代中か、交代不可か など。
- **ポケモン数値:** 存在フラグ・自分か・場にいるか・ひんしか・HP 割合・判明済みか・メガシンカ済みか・レベル、状態異常、能力ランク、タイプ、種族値。
- **ポケモン ID:** 種族・道具・特性・技 4 つの ID（モデル側で埋め込みベクトルに変換）。道具と特性は poke-env に ID 表が無いためハッシュで 1024 バケットに割り当て。0 は「不明」。
- **技数値:** 威力・命中・残り PP 割合・優先度・相手へのタイプ相性・タイプ一致・回復/吸収/反動・今使えるか、分類（物理/特殊/変化）、タイプ。
- Champions 対応の工夫:
  - `champions_max_pp()` / `pp_fraction()`: poke-env は本家ルールの最大 PP で数えるので、使用回数から Champions ルールでの PP 残量に換算。
  - `effective_species()`: 相手がメガシンカしても poke-env は種族名を元のまま保持するため、種族値からメガフォルムを逆算。
- `split_obs()` はベクトルをブロックごとのテンソルに戻す関数で、モデルが使います。

### `env.py` — 強化学習環境

- `RewardConfig`: 報酬の係数（victory / fainted / hp / status）。
- `PokemonEnv`: poke-env の `SinglesEnv` を継承し、観測を `encoding.encode_battle`、報酬を `RewardConfig` に基づく計算に差し替えたもの。
- `OpponentMixEnv`: 1 エージェント視点の Gymnasium ラッパー。`reset()` のたびに `mix` の重みで対戦相手を抽選し、試合終了時に勝敗・ターン数・相手名を `info` に入れて学習側に返します。
  `set_opponent_mix()` で学習中にカリキュラムや self-play プールを差し替えられます。
- `EnvConfig` / `make_env()`: 設定から環境を作る関数（並列環境の各サブプロセス内で実行される）。
  万一不正な行動が来てもクラッシュせずランダムな合法手に置き換えるよう `strict=False` で動かしています。

### `model.py` — Actor-Critic ネットワーク

1. 技ごとに「技 ID の埋め込み＋技の数値特徴」を MLP でトークン化。
2. ポケモンごとに「種族・道具・特性の埋め込み＋数値特徴＋そのポケモンの技トークンの平均」を MLP でトークン化（12 体分）。
3. 場の状態（GLOBAL）を 1 トークンにし、計 13 トークンを Transformer エンコーダに通して互いの関係を学習（存在しないスロットはマスク）。
4. **ポインタ型の行動ヘッド:** 各行動を「その行動が指すトークン」から採点します。
   - 交代 i → 自分のポケモン i のトークン＋全体文脈
   - 技 j → 場にいるポケモンの技 j のトークン＋そのポケモン＋全体文脈
   - メガシンカ＋技 j → 技 j のスコア＋ギミック用ヘッドの補正
5. 使えない行動のロジットを −1e9 にしてから softmax（マスク）。価値ヘッドは全体文脈トークンから勝ちやすさを予測。

`get_action_and_value()` が行動のサンプリング（`deterministic` なら argmax）・対数確率・エントロピー・価値をまとめて返します。
`save_checkpoint()` / `load_checkpoint()` は、モデルの重みと `ModelConfig`（＋任意でオプティマイザ状態・ステップ数・学習設定）を `.pt` に保存/復元します。

### `opponents.py` — 対戦相手

- `BASELINES`: `random` / `max_power` / `heuristic` の poke-env 組み込みプレイヤー。
- `PolicyPlayer`: 学習済みモデルで行動を選ぶ poke-env の `Player`。self-play の相手としても評価対象としても同じ推論経路を使います。1 手あたりの推論時間も計測。
- `OpponentFactory`: 名前（`random`, `heuristic`, `latest`, `pool`, `ckpt:<path>` など）から対戦相手を生成・キャッシュ。プールから外れたモデルはメモリから解放します。
  環境内の相手は `choose_move()` を呼ばれるだけなので、サーバに接続しない（オフラインの）プレイヤーとして作ります。
- `make_policy_player()`: チェックポイントから、サーバに接続する評価用の `PolicyPlayer` を作ります。

### `ppo.py` — PPO 学習

CleanRL 風の 1 ファイル完結の PPO 実装です。
`TrainConfig`（YAML の読み込み、ステップ数に応じたカリキュラムを返す `mix_at()`）、`resolve_device()`、学習ループ本体の `train()`、CLI の `main()` から成ります。
処理の流れは [5 章「どうやって実現しているか」](#どうやって実現しているか学習の流れ) を参照してください。

### `evaluate.py` — 評価

- `benchmark`: 1 つのチェックポイントを各ベースラインと `-n` 戦ずつ対戦させ、勝率・95% Wilson 信頼区間・平均ターン数・1 手あたり推論時間を出します。
  `--concurrency`（同時対戦数、既定 8）、`--deterministic`（サンプリングせず最大確率の行動を選ぶ）、`--opponents`（相手の指定）などのオプションがあります。
- `round_robin`: 複数のエージェント（チェックポイントやベースライン）で総当たりし、Bradley-Terry モデルで Elo 風レーティングを推定（`random` を 1000 に固定）。
- 学習時と違い Gymnasium 環境は使わず、poke-env の `Player` 同士を `battle_against()` で Showdown サーバ上で直接対戦させます。

---

## 未対応（Phase 1 の残タスク）

- 履歴を扱う LSTM/GRU 版（現在のモデルは「これまでに判明した情報の累積」しか見ていない）
- W&B ロギング（今は TensorBoard のみ）
- スループットの改善（M2・16 環境で約 430 ステップ/秒。ボトルネックは Python 側）
