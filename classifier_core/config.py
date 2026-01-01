from dataclasses import dataclass

@dataclass
class DatasetConfig:
    data_dir: str = "pubmed_rct"
    batch_size: int = 32
    max_tokens: int = 68000
    output_seq_len: int = 55 # Approx 95th percentile, can be auto-calculated but good to have default

@dataclass
class ModelConfig:
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 128
    dropout_rate: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 0.001
    model_save_dir: str = "classifier_core"
    model_name: str = "transformer_model.keras"
