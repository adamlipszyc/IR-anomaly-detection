from keras.models import Model
from keras.layers import Input, Dense

class DeepSVDD:
    def __init__(self, input_dim, latent_dim=32, kernel_initializer='glorot_uniform'):
        """
        Initialize Deep SVDD model.
        
        Args:
        - input_dim (int): Dimension of input data.
        - latent_dim (int): Number of latent features (encoding dimension).
        - kernel_initializer (str): Initializer for the weights.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kernel_initializer = kernel_initializer
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the Deep SVDD model.
        
        Returns:
        - model (keras.Model): The Deep SVDD model.
        """
        inputs = Input(shape=(self.input_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.latent_dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, x)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model

    def fit(self, data, epochs=50, batch_size=32):
        """
        Train the Deep SVDD model on the data.
        
        Args:
        - data (np.ndarray): The data to train on.
        - epochs (int): Number of epochs to train.
        - batch_size (int): Batch size for training.
        """
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size)

    def predict(self, data):
        """
        Predict using the Deep SVDD model.
        
        Args:
        - data (np.ndarray): The data for prediction.
        
        Returns:
        - predictions (np.ndarray): Predictions from the model.
        """
        return self.model.predict(data)
