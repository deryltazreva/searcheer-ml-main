import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import pickle

class JobCompatibilityNeuralNetwork:
    def __init__(self, max_features=10000, max_length=500, embedding_dim=128):
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def build_model(self):
        text_input = Input(shape=(self.max_length,), name='text_input')
        embedding = Embedding(self.max_features, self.embedding_dim)(text_input)
        lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
        pooling = GlobalMaxPooling1D()(lstm)
        text_dense = Dense(64, activation='relu')(pooling)
        text_dropout = Dropout(0.3)(text_dense)

        numerical_input = Input(shape=(8,), name='numerical_input')
        numerical_dense = Dense(32, activation='relu')(numerical_input)
        numerical_dropout = Dropout(0.2)(numerical_dense)

        combined = Concatenate()([text_dropout, numerical_dropout])
        combined_dense = Dense(128, activation='relu')(combined)
        combined_dropout = Dropout(0.3)(combined_dense)
        output = Dense(1, activation='sigmoid')(combined_dropout)

        self.model = Model(inputs=[text_input, numerical_input], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', 'mae'])
        return self.model

    def prepare_text_data(self, texts):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_length)

    def extract_numerical_features(self, cv_text, job_text):
        cv_words = set(cv_text.lower().split())
        job_words = set(job_text.lower().split())
        common_words = len(cv_words.intersection(job_words))

        features = [
            min(len(cv_words) / max(len(job_words), 1), 2.0),
            min(common_words / max(len(job_words), 1), 1.0)
        ]
        features += [0.5] * 6  #dummy
        return np.array(features)

    def train(self, training_data):
        combined_texts = list(training_data['cv_text']) + list(training_data['job_text'])
        X_text = self.prepare_text_data(combined_texts)
        n = len(training_data)
        X_cv = X_text[:n]
        X_job = X_text[n:]
        X_combined = np.concatenate([X_cv, X_job], axis=1)[:, :self.max_length]
        if X_combined.shape[1] < self.max_length:
            pad = np.zeros((X_combined.shape[0], self.max_length - X_combined.shape[1]))
            X_combined = np.concatenate([X_combined, pad], axis=1)

        X_numerical = [self.extract_numerical_features(row['cv_text'], row['job_text']) for _, row in training_data.iterrows()]
        X_numerical = self.scaler.fit_transform(np.array(X_numerical))

        y = training_data['label'].values

        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_combined, X_numerical, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        self.build_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
        ]

        self.model.fit(
            [X_text_train, X_num_train], y_train,
            validation_data=([X_text_test, X_num_test], y_test),
            epochs=20, batch_size=32, callbacks=callbacks, verbose=1
        )

        self.is_trained = True
        self.model.save("model/capstone.h5")
        with open("model/tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

    def predict(self, cv_text, job_text):
        if self.model is None or self.tokenizer is None:
            return 0.0
        combined_text = cv_text + " " + job_text
        seq = self.tokenizer.texts_to_sequences([combined_text])
        padded = pad_sequences(seq, maxlen=self.max_length)
        numerical = self.extract_numerical_features(cv_text, job_text).reshape(1, -1)
        numerical = self.scaler.transform(numerical)
        prediction = self.model.predict([padded, numerical], verbose=0)[0][0]
        return min(max(float(prediction), 0.0), 1.0) * 100
        
