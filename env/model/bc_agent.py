import logging
import numpy as np

import tensorflow as tf
from tensorflow import keras

logging.getLogger().setLevel(logging.INFO)


TRAINING_PARAMS = {
    "epochs": 500,
    "validation_split": 0.15,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "use_class_weights": False,
}

# TODO: be sure to sync up with action space on both ends
idx2action = ["U", "D", "L", "R", "I", "W"]

# ===================================== Utils  ==============================================


def load_data():
    # TODO :map
    #       player position
    #       remaining order position
    #       action
    return np.zeros((100, 100)), np.zeros((100,))


# ===================================== Model  ==============================================


def build_model(input_shape, action_shape):
    inputs = keras.Input(shape=input_shape[0], name="Overcooked_observation")
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(action_shape[0], name="targets")(x)
    return keras.Model(inputs=[inputs], outputs=outputs)


def train_model(verbose=False):
    inputs, targets = load_data()
    action_shape = [
        6,
    ]

    # ======================= build =============================

    model = build_model(inputs.shape, action_shape)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["sparse_categorical_accuracy"]
    model.compile(
        optimizer=keras.optimizers.Adam(TRAINING_PARAMS["learning_rate"]),
        loss=loss,
        metrics=metrics,
    )

    # ======================= train =============================

    inputs = {"Overcooked_observation": inputs}
    targets = {"targets": targets}

    batch_size = TRAINING_PARAMS["batch_size"]
    validation_split = TRAINING_PARAMS["validation_split"]
    epochs = TRAINING_PARAMS["epochs"]
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./bc_model_logs")
    model.fit(
        inputs,
        targets,
        callbacks=[tensorboard_callback],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=2 if verbose else 0,
    )

    # ======================= save =============================
    save_model(model)

    return model


def test_model():
    pass


def save_model(model, model_dir="./bc_model"):
    model.save(model_dir, save_format="tf")


def load_model(model_dir="./bc_model"):
    return keras.models.load_model(model_dir, custom_objects={"tf": tf})


# ===================================== Agent for Environment ==================================


class BC_Agent:
    def __init__(self):
        self.model = load_model()

    def action(self, states, return_logits=False, stochastic=True):
        logits = self.model.predict(states)
        if return_logits:
            return logits
        else:

            def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)  # only difference

            if stochastic:
                action_idx = np.random.choice(6, p=softmax(logits[0]))
            else:
                action_idx = np.argmax(logits)
            return idx2action[action_idx]


# ===================================== Unit test ==============================================


def __test_build_model():
    model = build_model(
        [12, 3],
        [
            4,
        ],
    )
    model.summary()


def __test_train_model():
    TRAINING_PARAMS["epochs"] = 3
    train_model(verbose=True)


def __test_agent_action():
    agent = BC_Agent()
    print(
        "test agent actin logits: ",
        agent.action(np.zeros((1, 100)), return_logits=True),
    )
    print("test agent action: ", agent.action(np.zeros((1, 100))))
    print("with idx 2 action:", idx2action)


if __name__ == "__main__":
    # __test_build_model()
    # __test_train_model()
    __test_agent_action()