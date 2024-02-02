from keras import Sequential
from keras.src.layers import Dense

# BUILDER PATTERN


class MlpModelBuilder:
    def __init__(self, trainer_configs):
        self.input_dim = trainer_configs.input_dim
        if self.input_dim is None:
            raise ValueError('You need to provide input_dim info')

    def build_mlp(self):
        """
        TODO: Write description

        """
        # Build the MLP model
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def set_params(self):
        pass


