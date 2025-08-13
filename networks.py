from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

class ActorController:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.actor = self.build_model()
        self.actor_target = self.build_model()

        self.actor_target.set_weights(self.actor.get_weights())

        self.optimizer = Adam(learning_rate=0.001, epsilon=0.1)

    def build_model(self, input_size=None):
        if input_size is None:
            input_size = self.state_size

        model = Sequential()
        model.add(Dense(256, input_dim=input_size, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(self.action_size, activation='tanh'))

        return model

class CriticController:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.critic = self.build_model()
        self.critic_target = self.build_model()

        self.critic_target.set_weights(self.critic.get_weights())

        self.optimizer = Adam(learning_rate=0.001, epsilon=0.1)

    def build_model(self, input_size=None):
        if input_size is None:
            input_size = self.state_size + self.action_size

        model = Sequential()
        model.add(Dense(512, input_dim=input_size, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(64, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(1, activation='linear'))

        return model

if __name__ == '__main__':
    import numpy as np

    actor = ActorController(state_size=4, action_size=2)
    actor_model = actor.build_model()
    inp = np.zeros((1,4), dtype=np.float32)
    inp[0,0] = 1
    p = actor_model.predict(inp)
    print(p)