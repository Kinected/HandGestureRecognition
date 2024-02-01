import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_PATH = "./datasets/distance_and_angles_gestures_dataset.csv"


dataset = pd.read_csv(DATASET_PATH)

labels = dataset["label"]
inputs = dataset.drop("label", axis=1)


inputs = inputs.to_numpy()
inputs = inputs.astype("float32")


dummies = pd.get_dummies(
    labels
)  # Creates a df of labels as categorical (ex : "closed" = [1, 0, 0, ..., 0])


print(dummies.isna().sum())
print(dummies.isnull().sum())

labels = dummies.to_numpy()


x_train, x_test, y_train, y_test = train_test_split(
    inputs, labels, test_size=0.2, random_state=42
)

from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

regularizer = regularizers.l2(0.001)

model = Sequential(
    [
        Input(shape=[42]),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(units=8, activation="softmax"),
    ]
)

optimizer = Adam(learning_rate=0.001)


model.summary()

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)


history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=1000,
    verbose=2,
    validation_split=0.2,
)
