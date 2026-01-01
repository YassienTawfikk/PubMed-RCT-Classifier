import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from classifier_core import data_loading, preprocessing, evaluation
from classifier_core.config import DatasetConfig
from tensorflow import keras

# Need to register custom layers for loading
from classifier_core.modeling import TransformerBlock, TokenAndPositionEmbedding

def run_pipeline(data: DatasetConfig, model_path: str = "classifier_core/transformer_model.keras"):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    # Load Data
    train_samples, _, test_samples = data_loading.load_data(data_dir=data.data_dir)
    
    # Process Data
    # 1. Prepare DataFrames
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    
    train_df = pd.DataFrame(train_samples)
    test_df = pd.DataFrame(test_samples)

    # 2. Fit Encoders on Training Data to ensure consistency
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(train_df["target"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))
    
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["target"].to_numpy())
    class_names = label_encoder.classes_
    
    # 3. Create Dataset (Raw text inputs, One-hot labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df["text"], test_labels_one_hot)).batch(data.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path, custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding
    })
    
    # Evaluate
    save_dir = os.path.dirname(model_path)
    evaluation.evaluate_model(model, test_dataset, class_names, save_dir=save_dir)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(run_pipeline)
