# Speech Emotion Recognition Project

This project aims to develop a machine learning model capable of recognizing human emotions from audio recordings of speech. We utilize the Speech Emotion dataset from Kaggle, which includes various spoken words/phrases with labeled emotions.

## Dataset

The dataset contains audio files of actors speaking phrases with different emotional intonations. The emotions included in the dataset are:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
- Calm

Each audio file is labeled with one of the above emotions.

## Preprocessing

We perform several preprocessing steps to prepare the data for our machine learning models:

- **Mel Spectrogram Conversion**: We convert audio signals into Mel spectrograms to capture the frequency and time domain information that is relevant for emotion recognition.
- **Feature Extraction**: We extract essential features from audio signals, including:
  - **Zero Crossing Rate (ZCR)**: Measures the number of times the signal crosses the zero line, indicating changes in the signal's energy.
  - **Root Mean Square Energy (RMSE)**: Indicates the power of the audio signal, which can reflect the speaker's emotional intensity.
  - **Mel-Frequency Cepstral Coefficients (MFCC)**: Captures the timbral aspects that are critical for identifying the emotion in speech.

## Data Augmentation

To mitigate the effects of class imbalance and enhance the diversity of our dataset, we apply various data augmentation techniques:

- **Adding Noise**: We introduce random noise to the audio samples to simulate real-world recording conditions.
- **Time Stretching**: We alter the speed of the audio clips without affecting the pitch.
- **Pitch Shifting**: We shift the pitch of the audio samples to represent a wider range of vocal pitches.

## Model Architecture

We employ a Convolutional Neural Network (CNN) for this classification task, with the following layers:

- Convolutional layers with ReLU activation
- Max pooling layers
- Dropout layers to prevent overfitting
- A final dense layer with a softmax activation function to output the probabilities of each emotion class

## Training

We use the following procedure to train our model:

- **Resampling**: To address class imbalance, we use techniques like SMOTE and ADASYN for oversampling the minority classes.
- **Model Compilation**: We compile our CNN with Adam optimizer and categorical crossentropy loss.
- **Metrics**: We track accuracy, precision, recall, and F1 score to evaluate model performance.
- **Validation**: We use a separate validation set to fine-tune hyperparameters and prevent overfitting.

## Results

The model performance is evaluated based on precision, recall, and F1 score metrics to account for class imbalance. We aim to achieve high accuracy across all emotion classes.

## Conclusion

This project demonstrates the use of audio signal processing and machine learning for the task of emotion recognition from speech. The approach can be extended and improved with additional data, more complex model architectures, or advanced feature extraction techniques.

## Bonus: Experimenting with Resampling Techniques

In addition to our primary approach, we experimented with various resampling techniques to further address the challenge of class imbalance. However, the following methods were not tested on the model because the model has already achieved high accuracy but you are welcome to try. The following techniques were explored:

### Upsampling the Minority Class

We duplicated instances from the minority class, such as 'calm', to match the prevalence of more common emotions. This method helped to balance the dataset but introduced the risk of overfitting due to the repetition of identical samples.

### Computing Class Weights

As an alternative to resampling, we explored the method of computing class weights to provide a solution to class imbalance. This technique assigns a higher weight to the minority class and a lower weight to the majority class, which allows the model to "pay more attention" to the underrepresented class during training.

### Synthetic Data Generation: SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) was used to create new, synthetic samples of the minority classes. By interpolating between existing samples, SMOTE added diversity to our training data without the exact repetition of instances.

### Adaptive Synthetic Sampling: ADASYN

Adaptive Synthetic Sampling (ADASYN) focuses on generating samples next to the borderline of decision, which are difficult for the model to learn. By doing so, it adapts to the data complexity and improves the model's performance on harder-to-learn instances.



