import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input, BatchNormalization, Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score
from keras.layers import GaussianNoise


class Plugin:
    """
    A predictor plugin using a simple LSTM network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'batch_size': 128,
        'intermediate_layers':3,
        'initial_layer_size': 32,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'dropout_rate': 0.1,
        'activation': 'tanh',
        'l2_reg': 1e-2,     # L2 regularization factor
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Build the LSTM model for multi-step time-series forecasting.

        Args:
            input_shape (tuple): Shape of the input data (time_steps, features).
        """
        self.params['input_dim'] = input_shape
        l2_reg = self.params.get('l2_reg', 1e-4)

        # Layer configuration
        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        layers.append(self.params['time_horizon'])  # Final output layer matches the time horizon

        # Debugging message
        print(f"LSTM Layer sizes: {layers}")
        print(f"LSTM input shape: {input_shape}")

        # Input shape: (time_steps, features)
        model_input = Input(shape=input_shape, name="model_input")  # Corrected input shape
        x = model_input
        
        # Add LSTM layers
        idx = 0
        for size in layers[:-1]:
            idx += 1
            if size > 1:
                x = LSTM(
                    units=size,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True,
                    name=f"lstm_layer_{idx}"
                )(x)
                      
        # Final LSTM layer without `return_sequences`
        x = LSTM(
            units=layers[-2],
            activation='tanh',
            recurrent_activation='sigmoid',
        )(x)

        x = BatchNormalization(name="batch_norm_final")(x)  # Shape: (batch_size, size)
        
        # Output layer
        model_output = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)
        
        #model_output = BatchNormalization()(model_output)

        # Build and compile the model
        self.model = Model(inputs=model_input, outputs=model_output, name="predictor_model")
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        self.model.compile(optimizer=adam_optimizer, loss=Huber(), metrics=['mse', 'mae'])
        #self.model.compile(optimizer=adam_optimizer, loss='mae', metrics=['mse', 'mae'])

        # Debugging messages
        print("Predictor Model Summary:")
        self.model.summary()




    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the LSTM model with optional validation data.
        """
        print(f"Training LSTM model with data shape: {x_train.shape}, target shape: {y_train.shape}")

        callbacks = []
        patience = self.params.get('patience', 14)
        use_daily = self.params.get('use_daily', False)

        # Early stopping based on validation loss
        early_stopping_monitor = EarlyStopping(
            monitor='loss', 
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)

        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None

        # Fit the model
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            #validation_data=validation_data,  # Provide validation data
            verbose=1,
            callbacks=callbacks,
            validation_split = 0.2
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Force the model to run in "training mode"
        preds_training_mode = self.model(x_train, training=True)
        mae_training_mode = np.mean(np.abs(preds_training_mode - y_train))
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        # Compare with evaluation mode
        preds_eval_mode = self.model(x_train, training=False)
        mae_eval_mode = np.mean(np.abs(preds_eval_mode - y_train))
        print(f"MAE in Evaluation Mode (manual): {mae_eval_mode:.6f}")

        # Evaluate on the full training dataset for consistency
        train_eval_results = self.model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        train_loss, train_mse, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MSE: {train_mse}, MAE: {train_mae}")
        
        val_eval_results = self.model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        val_loss, val_mse, val_mae = val_eval_results
        
        # Predict validation data for evaluation
        train_predictions = self.predict(x_train)  # Predict train data
        val_predictions = self.predict(x_val)      # Predict validation data

        # Calculate R² scores
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions



    def predict(self, data):
        print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        print(f"Predicted data shape: {predictions.shape}")
        return predictions

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with multi-step shape (N, time_horizon).
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        mse = np.mean((y_true_f - y_pred_f) ** 2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae


    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Predictor model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.build_model(input_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
