import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import optuna

dataset_root = "dataset"

# Global pipeline settings
WINDOW_SIZE = 50
STEP = 10
MOVING_AVG_KERNEL = 13     # kernel size for FEDformer's trend/seasonal decomposition
D_MODEL = 32               # internal embedding dimension used by FEDformer
N_MODES = 16               # number of random Fourier modes kept by each frequency block
AUTOENCODER_EPOCHS = 20
CNN_EPOCHS = 20
FEDFORMER_EPOCHS = 15
BATCH_SIZE = 16


# ________________________________________________________________________________
# DATASET INSPECTION / LOADING (unchanged logic from the original project)
# ________________________________________________________________________________

def show_dataset_structure():
    dataset_dirs = sorted(path for path in glob.glob(os.path.join(dataset_root, "*"))
                           if os.path.isdir(path))
    print("=" * 60)
    print("                     Dataset Structure")
    print("=" * 60)
    print(f"number of datasets: {len(dataset_dirs)}")

    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        dataset_name = os.path.basename(dataset_dir)
        csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        train_files = [file for file in csv_files if "train_normal" in os.path.basename(file)]
        test_files = [file for file in csv_files if "test_anormal" in os.path.basename(file)]
        print(f"\n {i}, {dataset_name}")
        print(f"    Normal training files: {len(train_files)}")
        print(f"    Anomalous test files: {len(test_files)}")


def inspect_all_experiment():
    dataset_dirs = sorted(path for path in glob.glob(os.path.join(dataset_root, "*"))
                           if os.path.isdir(path))
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            print("No CSV file found.")
            continue
        train_files = [file for file in csv_files if "train_normal" in os.path.basename(file)]
        test_files = [file for file in csv_files if "test_anormal" in os.path.basename(file)]

        print("\n" + "=" * 60)
        print(f"            DATASET: {dataset_name}")
        print("=" * 60)
        print(f"Total CSV files: {len(csv_files)}")
        print(f"Normal Training files: {len(train_files)}")
        print(f"Anomalous testing files: {len(test_files)}")
        if not test_files:
            continue

        df = pd.read_csv(test_files[0])
        print(f"\n Shape: {df.shape}")
        print("\n Columns:")
        print(df.columns.tolist())

        time_columns = [column for column in df.columns if column.lower() == "time"]
        label_columns = [column for column in df.columns if "label" in column.lower()]
        feature_columns = [column for column in df.columns if column not in time_columns + label_columns]

        print(f"\n Time columns: {time_columns}")
        print(f"\n Label columns : {label_columns}")
        print(f"\n Number of features: {len(feature_columns)}")
        print(f"\n Feature columns: {feature_columns}")

        if label_columns:
            label = label_columns[0]
            print("\n Label Distribution:")
            print(df[label].value_counts())
            print("\n Label Distribution(%):")
            print(df[label].value_counts(normalize=True) * 100)

        print("\n First 5 rows:")
        print(df.head())
        print("\n Data type:")
        print(df.dtypes)
        print("\n Missing values:")
        print(df.isnull().sum())

        if time_columns and label_columns and feature_columns:
            time = time_columns[0]
            label = label_columns[0]
            feature = feature_columns[0]
            plt.figure(figsize=(12, 5))
            plt.plot(df[time], df[feature], label=feature)
            anomaly_points = df[df[label] == 1]
            plt.scatter(anomaly_points[time], anomaly_points[feature], label="Anomaly")
            plt.xlabel("Time")
            plt.ylabel(feature)
            plt.title(f"{feature} over Time")
            plt.legend()
            plt.grid()
            os.makedirs("result_plots", exist_ok=True)
            plt.savefig(f"result_plots/{dataset_name}_{feature}.png", dpi=300, bbox_inches="tight")
            plt.close()


def load_normal_data(dataset_dir):
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    train_files = [file for file in csv_files if "train_normal" in os.path.basename(file)]
    dataframes = [pd.read_csv(file) for file in train_files]
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()


def load_test_data(dataset_dir):
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    test_files = [file for file in csv_files if "test_anormal" in os.path.basename(file).lower()]
    dataframes = [pd.read_csv(file) for file in test_files]
    if len(dataframes) == 0:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)


def prepare_features(df, feature_columns=None):
    time_columns = [column for column in df.columns if column.lower() == "time"]
    label_columns = [column for column in df.columns if "label" in column.lower()]
    if feature_columns is None:
        feature_columns = [column for column in df.columns if column not in time_columns + label_columns]
    X = df[feature_columns]
    return X, feature_columns


def creat_windows(data, window_size=50, step=10):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows, dtype=np.float32)


def create_label_windows(labels, window_size=50, step=10):
    window_labels = []
    for i in range(0, len(labels) - window_size + 1, step):
        window = labels[i:i + window_size]
        window_labels.append(1 if window.max() == 1 else 0)
    return np.array(window_labels)


# ________________________________________________________________________________
# BASELINE MODELS
# ________________________________________________________________________________

def build_autoencoder(window_size, n_features):
    inputs = keras.Input(shape=(window_size, n_features))
    x = layers.LSTM(64, activation="relu", return_sequences=True)(inputs)
    x = layers.LSTM(16, activation="relu", return_sequences=False)(x)
    encoded = x
    x = layers.RepeatVector(window_size)(encoded)
    x = layers.LSTM(16, activation="relu", return_sequences=True)(x)
    x = layers.LSTM(64, activation="relu", return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = keras.Model(inputs, outputs, name="LSTM_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn_autoencoder(window_size, n_features):
    inputs = keras.Input(shape=(window_size, n_features))
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling1D(2, padding="same")(x)

    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)

    x = layers.Cropping1D((0, x.shape[1] - window_size))(x) if x.shape[1] > window_size else x
    outputs = layers.Conv1D(n_features, kernel_size=3, activation="linear", padding="same")(x)
    model = keras.Model(inputs, outputs, name="CNN_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn_autoencoder_tunable(trial, window_size, n_features):
    filters_1 = trial.suggest_categorical("filters_1", [16, 32, 64])
    filters_2 = trial.suggest_categorical("filters_2", [8, 16, 32])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    inputs = keras.Input(shape=(window_size, n_features))
    x = layers.Conv1D(filters_1, kernel_size, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(filters_2, kernel_size, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling1D(2, padding="same")(x)

    x = layers.Conv1D(filters_2, kernel_size, activation="relu", padding="same")(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters_1, kernel_size, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Cropping1D((0, x.shape[1] - window_size))(x) if x.shape[1] > window_size else x
    outputs = layers.Conv1D(n_features, kernel_size, activation="linear", padding="same")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


# ________________________________________________________________________________
# FULL FEDformer IMPLEMENTATION
#
# Reference architecture (Zhou et al., 2022, "FEDformer: Frequency Enhanced
# Decomposed Transformer for Long-term Series Forecasting"), adapted here as an
# autoencoder for anomaly detection: instead of forecasting future steps, the
# model reconstructs its own input window, and reconstruction error is used as
# the anomaly score (same idea as the LSTM/CNN autoencoders above).
#
# Components implemented below:
#   1) SeriesDecomposition   - moving-average trend/seasonal decomposition
#   2) FrequencyEnhancedBlock (FEB-f) - self "attention" replacement: transforms
#      a random, fixed subset of Fourier modes with learnable complex weights
#   3) FrequencyEnhancedAttention (FEA) - cross-attention between decoder and
#      encoder computed in the frequency domain over a random subset of modes
#   4) Encoder/Decoder layers that stack the blocks above with residual
#      decomposition, exactly mirroring the paper's encoder/decoder structure
#   5) build_fedformer_autoencoder_full() - assembles the full model
# ________________________________________________________________________________

@keras.utils.register_keras_serializable(package="fedformer")
class SeriesDecomposition(layers.Layer):
    """Splits a series into a smooth trend (moving average) and a seasonal
    (residual) component, as in FEDformer / Autoformer."""

    def __init__(self, kernel_size=13, **kwargs):
        super().__init__(**kwargs)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        self.pad_left = pad
        self.pad_right = kernel_size - 1 - pad
        self.avg_pool = layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding="valid")

    def call(self, x):
        front = tf.repeat(x[:, :1, :], self.pad_left, axis=1)
        end = tf.repeat(x[:, -1:, :], self.pad_right, axis=1)
        x_padded = tf.concat([front, x, end], axis=1)
        trend = self.avg_pool(x_padded)
        seasonal = x - trend
        return seasonal, trend

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@keras.utils.register_keras_serializable(package="fedformer")
class FrequencyEnhancedBlock(layers.Layer):
    """FEB-f: replaces self-attention. Projects the series to the frequency
    domain (rFFT), keeps a fixed random subset of low/high frequency modes,
    applies a learnable complex-valued linear transform to each kept mode,
    zeroes out the rest, and maps back to the time domain (irFFT)."""

    def __init__(self, d_model, seq_len, n_modes=16, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        n_freq = seq_len // 2 + 1
        self.n_modes = int(min(n_modes, n_freq))
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.mode_indices = np.sort(rng.choice(n_freq, size=self.n_modes, replace=False)).astype(np.int64)

    def build(self, input_shape):
        init = keras.initializers.RandomUniform(-0.02, 0.02)
        self.w_real = self.add_weight(shape=(self.n_modes, self.d_model, self.d_model),
                                       initializer=init, trainable=True, name="feb_w_real")
        self.w_imag = self.add_weight(shape=(self.n_modes, self.d_model, self.d_model),
                                       initializer=init, trainable=True, name="feb_w_imag")
        self._mode_indices_tf = tf.constant(self.mode_indices.reshape(-1, 1), dtype=tf.int64)
        super().build(input_shape)

    def call(self, x):
        n_freq = self.seq_len // 2 + 1
        x_ft = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))       # (B, D, n_freq)
        x_ft = tf.transpose(x_ft, [0, 2, 1])                    # (B, n_freq, D)
        selected = tf.gather(x_ft, self.mode_indices, axis=1)   # (B, M, D) complex

        real = tf.math.real(selected)
        imag = tf.math.imag(selected)
        out_real = tf.einsum('bmd,mdo->bmo', real, self.w_real) - tf.einsum('bmd,mdo->bmo', imag, self.w_imag)
        out_imag = tf.einsum('bmd,mdo->bmo', real, self.w_imag) + tf.einsum('bmd,mdo->bmo', imag, self.w_real)
        out_complex = tf.complex(out_real, out_imag)            # (B, M, D)

        def scatter_one(o):
            return tf.scatter_nd(self._mode_indices_tf, o, shape=[n_freq, self.d_model])

        full_complex = tf.map_fn(scatter_one, out_complex, fn_output_signature=tf.complex64)
        full_complex = tf.transpose(full_complex, [0, 2, 1])
        x_time = tf.signal.irfft(full_complex, fft_length=[self.seq_len])
        x_time = tf.transpose(x_time, [0, 2, 1])
        return x_time

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "seq_len": self.seq_len,
                        "n_modes": self.n_modes, "seed": self.seed})
        return config


@keras.utils.register_keras_serializable(package="fedformer")
class FrequencyEnhancedAttention(layers.Layer):
    """FEA: cross-attention computed in the frequency domain between decoder
    queries and encoder keys/values, over fixed random subsets of modes on
    each side. Scores are the real part of the complex dot-product between
    query and key modes (a frequency-domain analogue of scaled dot-product
    attention); the resulting weighted value modes get one more learnable
    complex transform before being mapped back to the time domain."""

    def __init__(self, d_model, seq_len_q, seq_len_kv, n_modes=16, seed=43, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        n_freq_q = seq_len_q // 2 + 1
        n_freq_kv = seq_len_kv // 2 + 1
        self.n_modes = int(min(n_modes, n_freq_q, n_freq_kv))
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.q_mode_indices = np.sort(rng.choice(n_freq_q, size=self.n_modes, replace=False)).astype(np.int64)
        self.kv_mode_indices = np.sort(rng.choice(n_freq_kv, size=self.n_modes, replace=False)).astype(np.int64)

    def build(self, input_shape):
        init = keras.initializers.RandomUniform(-0.02, 0.02)
        self.w_real = self.add_weight(shape=(self.n_modes, self.d_model, self.d_model),
                                       initializer=init, trainable=True, name="fea_w_real")
        self.w_imag = self.add_weight(shape=(self.n_modes, self.d_model, self.d_model),
                                       initializer=init, trainable=True, name="fea_w_imag")
        self._q_mode_indices_tf = tf.constant(self.q_mode_indices.reshape(-1, 1), dtype=tf.int64)
        super().build(input_shape)

    def call(self, inputs):
        q, kv = inputs
        n_freq_q = self.seq_len_q // 2 + 1

        q_ft = tf.transpose(tf.signal.rfft(tf.transpose(q, [0, 2, 1])), [0, 2, 1])
        kv_ft = tf.transpose(tf.signal.rfft(tf.transpose(kv, [0, 2, 1])), [0, 2, 1])

        q_sel = tf.gather(q_ft, self.q_mode_indices, axis=1)     # (B, M, D) complex
        kv_sel = tf.gather(kv_ft, self.kv_mode_indices, axis=1)  # (B, M, D) complex

        qr, qi = tf.math.real(q_sel), tf.math.imag(q_sel)
        kr, ki = tf.math.real(kv_sel), tf.math.imag(kv_sel)

        # Real part of the complex dot-product <q, conj(kv)> summed over d_model,
        # used as an attention compatibility score between mode i (query) and
        # mode j (key/value).
        scores = tf.einsum('bid,bjd->bij', qr, kr) + tf.einsum('bid,bjd->bij', qi, ki)
        scores = scores / tf.sqrt(tf.cast(self.d_model, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)                 # (B, Mq, Mkv)

        out_real = tf.einsum('bij,bjd->bid', weights, kr)
        out_imag = tf.einsum('bij,bjd->bid', weights, ki)

        # one more learnable complex transform per (query) mode
        proj_real = tf.einsum('bmd,mdo->bmo', out_real, self.w_real) - tf.einsum('bmd,mdo->bmo', out_imag, self.w_imag)
        proj_imag = tf.einsum('bmd,mdo->bmo', out_real, self.w_imag) + tf.einsum('bmd,mdo->bmo', out_imag, self.w_real)
        out_complex = tf.complex(proj_real, proj_imag)

        def scatter_one(o):
            return tf.scatter_nd(self._q_mode_indices_tf, o, shape=[n_freq_q, self.d_model])

        full_complex = tf.map_fn(scatter_one, out_complex, fn_output_signature=tf.complex64)
        full_complex = tf.transpose(full_complex, [0, 2, 1])
        x_time = tf.signal.irfft(full_complex, fft_length=[self.seq_len_q])
        x_time = tf.transpose(x_time, [0, 2, 1])
        return x_time

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "seq_len_q": self.seq_len_q,
                        "seq_len_kv": self.seq_len_kv, "n_modes": self.n_modes, "seed": self.seed})
        return config


class FEDformerEncoderLayer(layers.Layer):
    """FEB (self) -> decomposition -> feed-forward -> decomposition.
    Only the seasonal part is propagated forward, matching the paper."""

    def __init__(self, d_model, seq_len, n_modes, moving_avg=13, ff_dim=None, dropout=0.05, seed=0, **kwargs):
        super().__init__(**kwargs)
        ff_dim = ff_dim or d_model * 2
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.feb = FrequencyEnhancedBlock(d_model, seq_len, n_modes, seed=42 + seed)
        self.ff = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)])
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        feb_out = self.feb(x)
        x = x + self.dropout(feb_out, training=training)
        seasonal1, _ = self.decomp1(x)

        ff_out = self.ff(seasonal1)
        x2 = seasonal1 + self.dropout(ff_out, training=training)
        seasonal2, _ = self.decomp2(x2)
        return seasonal2


class FEDformerDecoderLayer(layers.Layer):
    """Self-FEB -> decomposition -> cross-FEA with encoder output ->
    decomposition -> feed-forward -> decomposition, accumulating the trend
    components split off at every stage (exactly as in the paper)."""

    def __init__(self, d_model, seq_len_dec, seq_len_enc, n_modes, moving_avg=13,
                 ff_dim=None, dropout=0.05, seed=0, **kwargs):
        super().__init__(**kwargs)
        ff_dim = ff_dim or d_model * 2
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.self_feb = FrequencyEnhancedBlock(d_model, seq_len_dec, n_modes, seed=101 + seed)
        self.cross_fea = FrequencyEnhancedAttention(d_model, seq_len_dec, seq_len_enc, n_modes, seed=202 + seed)
        self.ff = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)])
        self.trend_proj = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, enc_out, training=False):
        self_out = self.self_feb(x)
        x = x + self.dropout(self_out, training=training)
        seasonal1, trend1 = self.decomp1(x)

        cross_out = self.cross_fea([seasonal1, enc_out])
        x2 = seasonal1 + self.dropout(cross_out, training=training)
        seasonal2, trend2 = self.decomp2(x2)

        ff_out = self.ff(seasonal2)
        x3 = seasonal2 + self.dropout(ff_out, training=training)
        seasonal3, trend3 = self.decomp3(x3)

        residual_trend = self.trend_proj(trend1 + trend2 + trend3)
        return seasonal3, residual_trend


def build_fedformer_autoencoder_full(window_size, n_features, d_model=D_MODEL, n_modes=N_MODES,
                                      encoder_layers=2, decoder_layers=1, moving_avg=MOVING_AVG_KERNEL,
                                      dropout=0.05, learning_rate=1e-3):
    """Assembles the complete FEDformer encoder-decoder as a reconstruction
    autoencoder: input window -> series decomposition -> frequency-enhanced
    encoder -> frequency-enhanced decoder (with cross attention) -> seasonal +
    trend projection -> reconstructed window."""

    if moving_avg >= window_size:
        moving_avg = window_size - 1
    if moving_avg % 2 == 0:
        moving_avg = max(3, moving_avg - 1)
    n_modes = max(2, min(n_modes, window_size // 2))

    inputs = keras.Input(shape=(window_size, n_features), name="fedformer_input")

    init_decomp = SeriesDecomposition(moving_avg, name="init_series_decomp")
    seasonal_init, trend_init = init_decomp(inputs)

    input_embed = layers.Dense(d_model, name="encoder_embedding")
    dec_embed = layers.Dense(d_model, name="decoder_seasonal_embedding")
    trend_embed = layers.Dense(d_model, name="decoder_trend_embedding")

    enc_x = input_embed(inputs)
    for i in range(encoder_layers):
        enc_x = FEDformerEncoderLayer(d_model, window_size, n_modes, moving_avg,
                                       dropout=dropout, seed=i, name=f"fed_encoder_{i}")(enc_x)

    dec_x = dec_embed(seasonal_init)
    trend_accum = trend_embed(trend_init)

    for i in range(decoder_layers):
        dec_x, residual_trend = FEDformerDecoderLayer(d_model, window_size, window_size, n_modes,
                                                        moving_avg, dropout=dropout, seed=i,
                                                        name=f"fed_decoder_{i}")(dec_x, enc_x)
        trend_accum = trend_accum + residual_trend

    seasonal_output = layers.Dense(n_features, name="seasonal_projection")(dec_x)
    trend_output = layers.Dense(n_features, name="trend_projection")(trend_accum)
    outputs = layers.Add(name="final_reconstruction")([seasonal_output, trend_output])

    model = keras.Model(inputs, outputs, name="FEDformer_Autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def objective(trial, X_windows, window_size):
    model = build_cnn_autoencoder_tunable(trial, window_size, X_windows.shape[2])
    history = model.fit(X_windows, X_windows, epochs=5, batch_size=BATCH_SIZE,
                         validation_split=0.1, verbose=0)
    return history.history["val_loss"][-1]


def evaluate_model(model, model_name, X_test_windows, y_test_windows, train_mse_ref=None):
    test_reconstructions = model.predict(X_test_windows, verbose=0)
    test_mse = np.mean(np.square(X_test_windows - test_reconstructions), axis=(1, 2))

    normal_test_mse = test_mse[y_test_windows == 0]
    if len(normal_test_mse) == 0:
        normal_test_mse = test_mse
    threshold = np.percentile(normal_test_mse, 95)

    y_pred = (test_mse > threshold).astype(int)
    print(f"Predicted anomaly ratio: {y_pred.mean():.2%}  |  Actual anomaly ratio: {y_test_windows.mean():.2%}")

    print("\n" + "=" * 60)
    print(f"EVALUATION: {model_name}")
    print("=" * 60)
    precision = precision_score(y_test_windows, y_pred, zero_division=0)
    recall = recall_score(y_test_windows, y_pred, zero_division=0)
    f1 = f1_score(y_test_windows, y_pred, zero_division=0)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    try:
        auc = roc_auc_score(y_test_windows, test_mse)
        print("ROC-AUC:", auc)
    except ValueError:
        auc = None
        print("ROC-AUC: can't evaluate, only one class present.")

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test_windows, y_pred))
    return {"model": model_name, "precision": precision, "recall": recall, "f1": f1,
            "roc_auc": auc, "test_mse": test_mse, "threshold": threshold}


# ________________________________________________________________________________
# MAIN PIPELINE - runs LSTM / CNN / full FEDformer autoencoders on every dataset
# ________________________________________________________________________________

def run_pipeline():
    all_results = []
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("result_plots", exist_ok=True)

    show_dataset_structure()
    dataset_dirs = sorted(path for path in glob.glob(os.path.join(dataset_root, "*"))
                           if os.path.isdir(path))

    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        print("\n" + "#" * 60)
        print(f"# DATASET: {dataset_name}")
        print("#" * 60)

        normal_df = load_normal_data(dataset_dir)
        if normal_df.empty:
            print(f"{dataset_name}: no normal training data found, skipping.")
            continue
        X, feature_columns = prepare_features(normal_df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_windows = creat_windows(X_scaled, window_size=WINDOW_SIZE, step=STEP)
        print(f"{dataset_name}: normal rows={len(X_scaled)} | train windows={X_windows.shape}")
        if X_windows.shape[0] == 0:
            print(f"{dataset_name}: not enough rows to build a full window, skipping.")
            continue

        test_df = load_test_data(dataset_dir)
        if test_df.empty:
            print(f"{dataset_name}: no test data found, skipping.")
            continue
        X_test, test_features = prepare_features(test_df, feature_columns=feature_columns)
        if len(test_features) == 0:
            print(f"{dataset_name}: no test features found, skipping.")
            continue
        X_test_scaled = scaler.transform(X_test)
        X_test_windows = creat_windows(X_test_scaled, window_size=WINDOW_SIZE, step=STEP)
        label_columns = [column for column in test_df.columns if "label" in column.lower()]
        if not label_columns or X_test_windows.shape[0] == 0:
            print(f"{dataset_name}: no usable labeled test windows, skipping.")
            continue
        y_test_windows = create_label_windows(test_df[label_columns[0]].values,
                                               window_size=WINDOW_SIZE, step=STEP)

        window_size = X_windows.shape[1]
        n_features = X_windows.shape[2]
        dataset_results = []

        # ---- 1) LSTM autoencoder (baseline) ----
        lstm_model = build_autoencoder(window_size, n_features)
        lstm_model.fit(X_windows, X_windows, epochs=AUTOENCODER_EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, shuffle=True, verbose=1)
        lstm_model.save(f"saved_models/{dataset_name}_lstm_autoencoder.keras")
        dataset_results.append(evaluate_model(lstm_model, "LSTM Autoencoder", X_test_windows, y_test_windows))

        # ---- 2) CNN autoencoder ----
        cnn_model = build_cnn_autoencoder(window_size, n_features)
        cnn_model.fit(X_windows, X_windows, epochs=CNN_EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.1, shuffle=True, verbose=1)
        cnn_model.save(f"saved_models/{dataset_name}_cnn_autoencoder.keras")
        dataset_results.append(evaluate_model(cnn_model, "CNN Autoencoder", X_test_windows, y_test_windows))

        # ---- 3) Full FEDformer autoencoder ----
        fedformer_model = build_fedformer_autoencoder_full(window_size, n_features)
        fedformer_model.summary()
        fedformer_model.fit(X_windows, X_windows, epochs=FEDFORMER_EPOCHS, batch_size=BATCH_SIZE,
                             validation_split=0.1, shuffle=True, verbose=1)
        fedformer_model.save(f"saved_models/{dataset_name}_fedformer_autoencoder.keras")
        dataset_results.append(evaluate_model(fedformer_model, "FEDformer Autoencoder", X_test_windows, y_test_windows))

        # ---- per-dataset comparison ----
        comparison_df = pd.DataFrame([{"Model": r["model"], "Precision": r["precision"], "Recall": r["recall"],
                                        "F1": r["f1"], "ROC-AUC": r["roc_auc"]} for r in dataset_results])
        print("\n" + "=" * 60)
        print(f"COMPARISON - {dataset_name}")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(f"result_plots/{dataset_name}_model_comparison.csv", index=False)

        plt.figure(figsize=(14, 5))
        colors = {"LSTM Autoencoder": "steelblue", "CNN Autoencoder": "seagreen", "FEDformer Autoencoder": "darkorange"}
        for r in dataset_results:
            plt.plot(r["test_mse"], label=f'{r["model"]} Reconstruction Error',
                      alpha=0.7, color=colors.get(r["model"]))
        true_anomaly_idx = np.where(y_test_windows == 1)[0]
        fed_result = dataset_results[-1]
        plt.axhline(y=fed_result["threshold"], color="red", linestyle="--", label="FEDformer Threshold")
        plt.scatter(true_anomaly_idx, fed_result["test_mse"][true_anomaly_idx],
                    color="black", marker="x", label="True Anomaly", zorder=5)
        plt.xlabel("Window Index")
        plt.ylabel("Reconstruction Error (MSE)")
        plt.title(f"LSTM vs CNN vs FEDformer - {dataset_name}")
        plt.legend()
        plt.grid()
        plt.savefig(f"result_plots/{dataset_name}_model_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()

        for r in dataset_results:
            all_results.append({"dataset": dataset_name, "model": r["model"], "precision": r["precision"],
                                 "recall": r["recall"], "f1": r["f1"],
                                 "roc_auc": r["roc_auc"] if r["roc_auc"] is not None else np.nan})

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - ALL DATASETS, ALL MODELS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv("final_results_summary.csv", index=False)

    if not results_df.empty:
        avg_df = (results_df.groupby("model")[["precision", "recall", "f1", "roc_auc"]]
                  .mean()
                  .reset_index()
                  .sort_values("f1", ascending=False)
                  .reset_index(drop=True))
        avg_df.insert(0, "rank", avg_df.index + 1)
        avg_df.columns = ["Rank", "Model", "Avg Precision", "Avg Recall", "Avg F1", "Avg ROC-AUC"]

        print("\n" + "=" * 60)
        print("AVERAGE PERFORMANCE PER MODEL (across all datasets) - BEST FIRST")
        print("=" * 60)
        print(avg_df.to_string(index=False))
        print(f"\nBest overall model (by average F1): {avg_df.iloc[0]['Model']}")

        avg_df.to_csv("model_average_comparison.csv", index=False)

    return results_df


if __name__ == "__main__":
    run_pipeline()
    inspect_all_experiment()
