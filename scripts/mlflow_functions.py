import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

#  for Decision Tree
def run_decision_tree(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(model, "DecisionTree_model")
        
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("criterion", model.criterion)

        print(f"Decision Tree accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

# for Random Forest
def run_random_forest(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(model, "RandomForest_model")
        
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", model.n_estimators)

        print(f"Random Forest accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

# for Gradient Boosting
def run_gradient_boosting(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(model, "GradientBoosting_model")
        
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("learning_rate", model.learning_rate)

        print(f"Gradient Boosting accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

#for MLP Classifier
def run_mlp_classifier(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = MLPClassifier(max_iter=300)
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(model, "MLP_model")
        
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("hidden_layer_sizes", model.hidden_layer_sizes)

        print(f"MLP Classifier accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

#for CNN
def run_cnn(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Modify input shape as per your data
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        y_train_one_hot = to_categorical(y_train, 10)  # Modify the number of classes
        y_test_one_hot = to_categorical(y_test, 10)

        model.fit(x_train, y_train_one_hot, epochs=10)
        loss, accuracy = model.evaluate(x_test, y_test_one_hot)

        mlflow.tensorflow.log_model(model, "CNN_model")
        mlflow.log_metric("accuracy", accuracy)

        print(f"CNN accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

#for RNN
def run_rnn(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = Sequential([
            SimpleRNN(128, input_shape=(28, 28)),  # Modify input shape as per your data
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        y_train_one_hot = to_categorical(y_train, 10)  # Modify the number of classes
        y_test_one_hot = to_categorical(y_test, 10)

        model.fit(x_train, y_train_one_hot, epochs=10)
        loss, accuracy = model.evaluate(x_test, y_test_one_hot)

        mlflow.tensorflow.log_model(model, "RNN_model")
        mlflow.log_metric("accuracy", accuracy)

        print(f"RNN accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

#for LSTM
def run_lstm(x_train, y_train, x_test, y_test):
    with mlflow.start_run():
        model = Sequential([
            LSTM(128, input_shape=(28, 28)),  # Modify input shape as per your data
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        y_train_one_hot = to_categorical(y_train, 10)  # Modify the number of classes
        y_test_one_hot = to_categorical(y_test, 10)

        model.fit(x_train, y_train_one_hot, epochs=10)
        loss, accuracy = model.evaluate(x_test, y_test_one_hot)

        mlflow.tensorflow.log_model(model, "LSTM_model")
        mlflow.log_metric("accuracy", accuracy)

        print(f"LSTM accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
def run_cnn_1d(x_train, y_train, x_test, y_test):
    # Reshape input to be compatible with Conv1D (samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy}")

# Call your functions here, for example:
# run_decision_tree(x_train_fraud, y_train_fraud, x_test_fraud, y_test_fraud)
