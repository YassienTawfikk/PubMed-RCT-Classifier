import sys
import os

# Add project root to path to allow importing classifier_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from classifier_core import downloader, data_loading, preprocessing, modeling, utils, evaluation
from classifier_core.config import DatasetConfig, ModelConfig, TrainConfig

def run_pipeline(data: DatasetConfig, model_cfg: ModelConfig, train: TrainConfig):
    # Check GPU
    utils.check_gpu()
    
    # Download Data
    downloader.download_pubmed_data(data_dir=data.data_dir)
    
    # Load Data
    train_samples, val_samples, test_samples = data_loading.load_data(data_dir=data.data_dir)
    
    # Preprocess & Create Datasets
    train_ds, val_ds, test_ds, text_vectorizer, class_names, output_seq_len = preprocessing.create_datasets(
        train_samples, val_samples, test_samples,
        batch_size=data.batch_size,
        max_tokens=data.max_tokens,
        output_seq_len=data.output_seq_len
    )
    
    print(f"Class names: {class_names}")
    
    # Build Model
    model = modeling.build_model(
        text_vectorizer=text_vectorizer,
        vocab_size=data.max_tokens,
        output_seq_len=output_seq_len,
        num_classes=len(class_names),
        embed_dim=model_cfg.embed_dim,
        num_heads=model_cfg.num_heads,
        ff_dim=model_cfg.ff_dim
    )
    
    model.summary()
    
    # Train Model
    print("Starting training...")
    history = model.fit(train_ds,
                        epochs=train.epochs,
                        validation_data=val_ds,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5)])
    
    # Save Model
    if not os.path.exists(train.model_save_dir):
        os.makedirs(train.model_save_dir)
        
    model_path = os.path.join(train.model_save_dir, train.model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot curves
    evaluation.plot_loss_curves(history, save_dir=train.model_save_dir)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(run_pipeline)
