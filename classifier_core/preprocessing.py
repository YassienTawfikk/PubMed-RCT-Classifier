import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.layers import TextVectorization

def create_datasets(train_samples, val_samples, test_samples, batch_size=32, max_tokens=68000, output_seq_len=None):
    """
    Preprocesses data and creating tf.data.Datasets.
    Returns:
        train_dataset, val_dataset, test_dataset (tf.data.Dataset)
        text_vectorizer (TextVectorization layer)
        class_names (list)
        output_seq_len (int)
    """
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)

    # --- Label Encoding ---
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
    val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

    # Get class names
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
    class_names = label_encoder.classes_

    # --- Text Vectorization ---
    if output_seq_len is None:
        sent_lens = [len(sentence.split()) for sentence in train_df["text"]]
        output_seq_len = int(np.percentile(sent_lens, 95))
    
    text_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_seq_len)
    
    # Adapt vectorizer
    print("Adapting TextVectorization layer...")
    text_vectorizer.adapt(train_df["text"].tolist())

    # --- Dataset Creation ---
    print("Creating tf.data.Datasets...")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df["text"], train_labels_one_hot)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_df["text"], val_labels_one_hot)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df["text"], test_labels_one_hot)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, text_vectorizer, class_names, output_seq_len
