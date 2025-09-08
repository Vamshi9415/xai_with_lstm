# Enhanced Spam Detection Pipeline with Multiple Models and Training Procedures
import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, SpatialDropout1D, 
    GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, 
    GlobalAveragePooling1D, Concatenate, Input, Flatten,
    BatchNormalization, Attention, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow as tf
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

class SpamDetectionPipeline:
    """
    Comprehensive spam detection pipeline with multiple model architectures
    and training procedures.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tokenizer = None
        self.label_encoder = None
        self.models = {}
        self.histories = {}
        self.results = {}
        
    # ==== DATA LOADING & PREPROCESSING ====
    
    def load_and_clean_data(self, path: str, text_col: str = 'Message', 
                           label_col: str = 'Category') -> pd.DataFrame:
        """Load and clean the dataset."""
        df = pd.read_csv(path)
        df = df[[label_col, text_col]].copy()
        df.columns = ['Category', 'Message']
        df = df.dropna().reset_index(drop=True)
        
        # Clean text
        df['Message'] = df['Message'].apply(self._clean_text)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['Category_encoded'] = self.label_encoder.fit_transform(df['Category'])
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_pad(self, df: pd.DataFrame, max_vocab_ratio: float = 0.15, 
                        max_length: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
        """Tokenize text and pad sequences."""
        total_word_count = df['Message'].apply(lambda x: len(x.split())).sum()
        max_vocab_size = math.floor(total_word_count * max_vocab_ratio)
        
        self.tokenizer = Tokenizer(
            num_words=max_vocab_size, 
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True
        )
        
        self.tokenizer.fit_on_texts(df['Message'])
        sequences = self.tokenizer.texts_to_sequences(df['Message'])
        
        X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        y = df['Category_encoded'].values
        
        vocab_size = min(len(self.tokenizer.word_index) + 1, max_vocab_size)
        
        return X, y, vocab_size
    
    # ==== MODEL ARCHITECTURES ====
    
    def build_lstm_single(self, vocab_size: int, max_length: int, 
                         embedding_dim: int = 128, lstm_units: int = 64,
                         dropout_rate: float = 0.2, regularizer=None) -> Model:
        """Single LSTM layer model."""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length,
                     embeddings_regularizer=regularizer),
            SpatialDropout1D(dropout_rate),
            LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate,
                 kernel_regularizer=regularizer, recurrent_regularizer=regularizer),
            Dense(32, activation='relu', kernel_regularizer=regularizer),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_lstm_stacked(self, vocab_size: int, max_length: int,
                          embedding_dim: int = 128, lstm_units: int = 64,
                          dropout_rate: float = 0.2, num_layers: int = 2,
                          regularizer=None) -> Model:
        """Stacked LSTM model."""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length,
                     embeddings_regularizer=regularizer),
            SpatialDropout1D(dropout_rate)
        ])
        
        # Add stacked LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            units = lstm_units // (i + 1) if i > 0 else lstm_units
            
            model.add(LSTM(
                units, 
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                return_sequences=return_sequences,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer
            ))
            
            if return_sequences:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizer))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        return model
    
    def build_bidirectional_lstm(self, vocab_size: int, max_length: int,
                                embedding_dim: int = 128, lstm_units: int = 64,
                                dropout_rate: float = 0.2, regularizer=None) -> Model:
        """Bidirectional LSTM model."""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length,
                     embeddings_regularizer=regularizer),
            SpatialDropout1D(dropout_rate),
            Bidirectional(LSTM(lstm_units, dropout=dropout_rate, 
                             recurrent_dropout=dropout_rate,
                             kernel_regularizer=regularizer,
                             recurrent_regularizer=regularizer)),
            Dense(64, activation='relu', kernel_regularizer=regularizer),
            Dropout(dropout_rate),
            Dense(32, activation='relu', kernel_regularizer=regularizer),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_gru_model(self, vocab_size: int, max_length: int,
                       embedding_dim: int = 128, gru_units: int = 64,
                       dropout_rate: float = 0.2, regularizer=None) -> Model:
        """GRU-based model."""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length,
                     embeddings_regularizer=regularizer),
            SpatialDropout1D(dropout_rate),
            GRU(gru_units, dropout=dropout_rate, recurrent_dropout=dropout_rate,
                kernel_regularizer=regularizer, recurrent_regularizer=regularizer),
            Dense(32, activation='relu', kernel_regularizer=regularizer),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_cnn_model(self, vocab_size: int, max_length: int,
                       embedding_dim: int = 128, filters: int = 128,
                       kernel_sizes: List[int] = [3, 4, 5],
                       dropout_rate: float = 0.2, regularizer=None) -> Model:
        """CNN model with multiple kernel sizes."""
        inputs = Input(shape=(max_length,))
        
        embedding = Embedding(vocab_size, embedding_dim, input_length=max_length,
                            embeddings_regularizer=regularizer)(inputs)
        embedding = SpatialDropout1D(dropout_rate)(embedding)
        
        conv_outputs = []
        for kernel_size in kernel_sizes:
            conv = Conv1D(filters, kernel_size, activation='relu',
                         kernel_regularizer=regularizer)(embedding)
            conv = GlobalMaxPooling1D()(conv)
            conv_outputs.append(conv)
        
        if len(conv_outputs) > 1:
            merged = Concatenate()(conv_outputs)
        else:
            merged = conv_outputs[0]
        
        dense = Dense(64, activation='relu', kernel_regularizer=regularizer)(merged)
        dense = Dropout(dropout_rate)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_lstm_hybrid(self, vocab_size: int, max_length: int,
                             embedding_dim: int = 128, filters: int = 64,
                             kernel_size: int = 3, lstm_units: int = 64,
                             dropout_rate: float = 0.2, regularizer=None) -> Model:
        """CNN-LSTM hybrid model."""
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length,
                     embeddings_regularizer=regularizer),
            SpatialDropout1D(dropout_rate),
            Conv1D(filters, kernel_size, activation='relu',
                   kernel_regularizer=regularizer),
            LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate,
                 kernel_regularizer=regularizer, recurrent_regularizer=regularizer),
            Dense(32, activation='relu', kernel_regularizer=regularizer),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_attention_model(self, vocab_size: int, max_length: int,
                             embedding_dim: int = 128, lstm_units: int = 64,
                             dropout_rate: float = 0.2, regularizer=None) -> Model:
        """LSTM with attention mechanism."""
        inputs = Input(shape=(max_length,))
        
        embedding = Embedding(vocab_size, embedding_dim, input_length=max_length,
                            embeddings_regularizer=regularizer)(inputs)
        embedding = SpatialDropout1D(dropout_rate)(embedding)
        
        lstm_out = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate,
                       recurrent_dropout=dropout_rate, 
                       kernel_regularizer=regularizer,
                       recurrent_regularizer=regularizer)(embedding)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(lstm_units)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        sent_representation = tf.keras.layers.multiply([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.sum(x, axis=1)
        )(sent_representation)
        
        dense = Dense(32, activation='relu', kernel_regularizer=regularizer)(sent_representation)
        dense = Dropout(dropout_rate)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    # ==== SAMPLING TECHNIQUES ====
    
    def apply_sampling_technique(self, X: np.ndarray, y: np.ndarray, 
                               technique: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
        """Apply various sampling techniques to handle class imbalance."""
        if technique == 'none':
            return X, y
        
        sampling_techniques = {
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
            'random_undersample': RandomUnderSampler(random_state=self.random_state),
            'smote_tomek': SMOTETomek(random_state=self.random_state),
            'smote_enn': SMOTEENN(random_state=self.random_state),
        }
        
        if technique not in sampling_techniques:
            raise ValueError(f"Unknown sampling technique: {technique}")
        
        sampler = sampling_techniques[technique]
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        return X_resampled, y_resampled
    
    # ==== CALLBACKS AND TRAINING UTILITIES ====
    
    def get_callbacks(self, model_name: str, monitor: str = 'val_loss',
                     patience: int = 15, reduce_lr_patience: int = 5,
                     min_lr: float = 1e-6, factor: float = 0.5) -> List:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_{model_name}.h5',
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def compile_model(self, model: Model, optimizer: str = 'adam',
                     learning_rate: float = 0.001, loss: str = 'binary_crossentropy',
                     metrics: List[str] = ['accuracy']) -> Model:
        """Compile model with specified parameters."""
        optimizers = {
            'adam': Adam(learning_rate=learning_rate),
            'rmsprop': RMSprop(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate)
        }
        
        model.compile(
            optimizer=optimizers.get(optimizer, Adam(learning_rate=learning_rate)),
            loss=loss,
            metrics=metrics
        )
        return model
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return dict(enumerate(class_weights))
    
    # ==== TRAINING PROCEDURES ====
    
    def train_single_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, model_name: str,
                          sampling_technique: str = 'none', use_class_weight: bool = False,
                          batch_size: int = 32, epochs: int = 100) -> Any:
        """Train a single model with specified configuration."""
        
        # Apply sampling technique
        X_train_sampled, y_train_sampled = self.apply_sampling_technique(
            X_train, y_train, sampling_technique
        )
        
        # Get class weights if requested
        class_weight = None
        if use_class_weight and sampling_technique == 'none':
            class_weight = self.get_class_weights(y_train_sampled)
        
        # Get callbacks
        callbacks = self.get_callbacks(model_name)
        
        # Train model
        history = model.fit(
            X_train_sampled, y_train_sampled,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return history
    
    def cross_validate_model(self, model_builder, X: np.ndarray, y: np.ndarray,
                           model_params: Dict, cv_folds: int = 5) -> Dict:
        """Perform cross-validation on a model."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nTraining fold {fold + 1}/{cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Build fresh model for each fold
            tf.keras.backend.clear_session()
            model = model_builder(**model_params)
            model = self.compile_model(model)
            
            # Train model
            history = self.train_single_model(
                model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                f"fold_{fold}", epochs=50
            )
            
            # Evaluate fold
            val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            cv_scores.append(val_acc)
            fold_histories.append(history)
            
        return {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'fold_scores': cv_scores,
            'histories': fold_histories
        }
    
    # ==== EVALUATION ====
    
    def evaluate_model(self, model: Model, X_test: np.ndarray, y_test: np.ndarray,
                      threshold: float = 0.5, model_name: str = "Model") -> Dict:
        """Comprehensive model evaluation."""
        # Predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        y_pred_prob = y_pred_prob.flatten()
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_prob
        }
        
        # Print results
        print(f"\n=== {model_name} Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)}")
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        
        return results
    
    # ==== MAIN PIPELINE ====
    
    def run_comprehensive_pipeline(self, data_path: str, test_size: float = 0.2,
                                 val_size: float = 0.2, max_length: int = 100,
                                 embedding_dim: int = 128) -> Dict:
        """Run comprehensive pipeline with multiple models and techniques."""
        
        print("Loading and preprocessing data...")
        df = self.load_and_clean_data(data_path)
        X, y, vocab_size = self.tokenize_and_pad(df, max_length=max_length)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), stratify=y_temp, 
            random_state=self.random_state
        )
        
        print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        # Model configurations
        model_configs = {
            'lstm_single': {
                'builder': self.build_lstm_single,
                'params': {'vocab_size': vocab_size, 'max_length': max_length, 
                          'embedding_dim': embedding_dim, 'lstm_units': 64}
            },
            'lstm_stacked': {
                'builder': self.build_lstm_stacked,
                'params': {'vocab_size': vocab_size, 'max_length': max_length,
                          'embedding_dim': embedding_dim, 'lstm_units': 64, 'num_layers': 2}
            },
            'bidirectional_lstm': {
                'builder': self.build_bidirectional_lstm,
                'params': {'vocab_size': vocab_size, 'max_length': max_length,
                          'embedding_dim': embedding_dim, 'lstm_units': 64}
            },
            'gru': {
                'builder': self.build_gru_model,
                'params': {'vocab_size': vocab_size, 'max_length': max_length,
                          'embedding_dim': embedding_dim, 'gru_units': 64}
            },
            'cnn': {
                'builder': self.build_cnn_model,
                'params': {'vocab_size': vocab_size, 'max_length': max_length,
                          'embedding_dim': embedding_dim, 'filters': 128}
            },
            'cnn_lstm': {
                'builder': self.build_cnn_lstm_hybrid,
                'params': {'vocab_size': vocab_size, 'max_length': max_length,
                          'embedding_dim': embedding_dim, 'filters': 64, 'lstm_units': 64}
            },
            'attention': {
                'builder': self.build_attention_model,
                'params': {'vocab_size': vocab_size, 'max_length': max_length,
                          'embedding_dim': embedding_dim, 'lstm_units': 64}
            }
        }
        
        # Training configurations
        sampling_techniques = ['none', 'smote', 'class_weight']
        
        all_results = {}
        
        # Train all model combinations
        for model_name, config in model_configs.items():
            for sampling in sampling_techniques:
                
                config_name = f"{model_name}_{sampling}"
                print(f"\n{'='*60}")
                print(f"Training {config_name}")
                print(f"{'='*60}")
                
                tf.keras.backend.clear_session()
                
                # Build and compile model
                model = config['builder'](**config['params'])
                model = self.compile_model(model)
                
                # Train model
                use_class_weight = (sampling == 'class_weight')
                sampling_tech = sampling if sampling != 'class_weight' else 'none'
                
                history = self.train_single_model(
                    model, X_train, y_train, X_val, y_val, config_name,
                    sampling_technique=sampling_tech, use_class_weight=use_class_weight
                )
                
                # Evaluate model
                results = self.evaluate_model(model, X_test, y_test, model_name=config_name)
                
                # Store results
                self.models[config_name] = model
                self.histories[config_name] = history
                all_results[config_name] = results
        
        self.results = all_results
        return all_results
    
    def plot_training_history(self, model_name: str):
        """Plot training history for a specific model."""
        if model_name not in self.histories:
            print(f"No history found for model: {model_name}")
            return
        
        history = self.histories[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        if not self.results:
            print("No results to compare. Run the pipeline first.")
            return None
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'AUC Score': results['auc_score'],
                'Precision (Spam)': results['classification_report']['1']['precision'],
                'Recall (Spam)': results['classification_report']['1']['recall'],
                'F1-Score (Spam)': results['classification_report']['1']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def save_pipeline(self, filepath: str):
        """Save the entire pipeline."""
        pipeline_data = {
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'results': self.results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a saved pipeline."""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.tokenizer = pipeline_data['tokenizer']
        self.label_encoder = pipeline_data['label_encoder']
        self.results = pipeline_data['results']
        
        print(f"Pipeline loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SpamDetectionPipeline(random_state=42)
    
    # Run comprehensive analysis
    results = pipeline.run_comprehensive_pipeline(
        'archive/SPAM text message 20170820 - Data.csv',
        test_size=0.2,
        val_size=0.2,
        max_length=100,
        embedding_dim=128
    )
    
    # Compare models
    comparison = pipeline.compare_models()
    print("\nModel Comparison:")
    print(comparison)
    
    # Plot training history for best model
    best_model = comparison.iloc[0]['Model']
    pipeline.plot_training_history(best_model)
    
    # Save pipeline
    pipeline.save_pipeline('spam_detection_pipeline.pkl')
    
    print("Pipeline ready! Uncomment the lines above to run the full analysis.")