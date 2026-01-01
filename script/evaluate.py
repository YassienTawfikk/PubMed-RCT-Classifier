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

    # Load Data (we only need test data really, but loading functions might grab all)
    # Optimally we should have a function to load just test, but `load_data` does all.
    _, _, test_samples = data_loading.load_data(data_dir=data.data_dir)
    
    # We need to recreate the dataset pipeline. 
    # Note: TextVectorization state is part of the saved model usually if included in the model itself.
    # IN modeling.py we passed text_vectorizer to the model, so it is a layer in the model.
    # However, to create the dataset input (text, label), we still need to process the text/labels same way.
    
    # Re-using create_datasets but ignoring vectorizer adaptation since model has it
    # Actually wait, `create_datasets` adapts vectorizer. We don't want to re-adapt on test data!
    # But since we are loading a model that HAS the vectorizer layer inside it, we just need raw text for inputs?
    # Let's check modeling.py: inputs = layers.Input(shape=(1,), dtype=tf.string) -> text_vectorizer(inputs)
    # Yes, the model accepts raw strings!
    # So we just need to one-hot encode the labels.
    
    # Let's manually do dataset creation for test to avoid re-adapting vectorizer
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    
    test_df = pd.DataFrame(test_samples)
    
    # We need to ensure label encoder matches training. Ideally we save the encoder. 
    # For now, assuming standard alphabetical order of classes as in training.
    # Ideally `load_data` returns consistent order if we re-run it.
    
    # Hack: We need to fit label encoder on something to know classes. 
    # Let's just load all data again to be safe about label mapping.
    train_samples, _, _ = data_loading.load_data(data_dir=data.data_dir)
    train_df = pd.DataFrame(train_samples)
    
    # Label Encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(train_df["target"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))
    
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["target"].to_numpy())
    class_names = label_encoder.classes_
    
    # Create Dataset (Source: raw text, label: one-hot)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df["text"], test_labels_one_hot)).batch(data.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Load Model with custom objects
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
