import numpy as np
import heapq 
import math
class trigram_model:
    def __init__(self, limit=5, model_type="code"):
        self.limit = limit
        self.type = model_type
        self.distribution = {}
        self.data = []  # This will hold the original training data for retraining purposes.
        self.train_set = []
        self.experiment_set = []

    def get_type(self):
        """
        return the type of data the model is trained on
        :return: type of data
        """
        return self.type
    def get_limit(self):
        """
        return the maximum number of suggestions the model can provide
        :return: suggestion limit 
        """
        return self.limit
    def train(self, data):
        """
        Trains the model using the provided list of tokens.
        :param data: List of tokens to train the model.
        """
        self.data = data
        unique_token_size = len(set(data))
        total_size = len(data)
        print(f"the {self.get_type()} model is trained based on {total_size} token(s)")
        print(f"the {self.get_type()} model is trained based on {unique_token_size} unique token(s)")
        experiment_size = int(total_size * 0.1)  # Size of the experimental set (10% of the total)

        # To ensure that the experiment set is a continuous chunk, we select a random starting index
        start_index = np.random.randint(0, total_size - experiment_size)

        # Define the end index for the experiment set
        end_index = start_index + experiment_size

        # Split the original data into training and experiment sets while maintaining continuity
        self.experiment_set = data[start_index:end_index]
        self.train_set = data[:start_index] + data[end_index:]

        # Now, you can proceed with the training on the self.train_set
        for i in range(len(self.train_set) - 2):
            # Create the trigram parts
            token1, token2, token3 = self.train_set[i], self.train_set[i + 1], self.train_set[i + 2]
            key = (token1, token2)

            # If the key already exists in the distribution, update the frequency of the third token
            if key in self.distribution:
                if token3 in self.distribution[key]:
                    self.distribution[key][token3] += 1  # Increment the count for the existing third token
                else:
                    self.distribution[key][token3] = 1  # Initialize the count for the new third token
            else:
                # If the key doesn't exist, create a new entry in the distribution
                self.distribution[key] = {token3: 1}
    def deterministic_train(self, data):
        """
        Trains the model using the provided list of tokens.
        :param data: List of tokens to train the model.
        """
        self.data = data
        unique_token_size = len(set(data))
        total_size = len(data)
        print(f"the {self.get_type()} model is trained based on {total_size} token(s)")
        print(f"the {self.get_type()} model is trained based on {unique_token_size} unique token(s)")
        experiment_size = int(total_size * 0.1)  # Size of the experimental set (10% of the total)
        # Split the original data into training and experiment sets while maintaining continuity
        self.experiment_set = data[:experiment_size]
        self.train_set = data[experiment_size:]

        # Now, you can proceed with the training on the self.train_set
        for i in range(len(self.train_set) - 2):
            # Create the trigram parts
            token1, token2, token3 = self.train_set[i], self.train_set[i + 1], self.train_set[i + 2]
            key = (token1, token2)

            # If the key already exists in the distribution, update the frequency of the third token
            if key in self.distribution:
                if token3 in self.distribution[key]:
                    self.distribution[key][token3] += 1  # Increment the count for the existing third token
                else:
                    self.distribution[key][token3] = 1  # Initialize the count for the new third token
            else:
                # If the key doesn't exist, create a new entry in the distribution
                self.distribution[key] = {token3: 1}

   
    def retrain(self):
        """
        Retrains the model with the original training data.
        the training and experiment set will be randomized
        """
        if self.data:
            self.train(self.data)

    def predict(self, tokens):
        """
        Predicts the next set of tokens based on the input.
        :param tokens: List of tokens to base the prediction on.
        :return: A list of predicted tokens.
        """

        if not isinstance(tokens, tuple) or len(tokens) != 2:
            raise ValueError("Input must be a tuple of exactly two tokens.")

        # Check if the token pair is in the distribution.
        if tuple(tokens) not in self.distribution:
            # print("Token sequence not found in the training data.")
            return []

        # Get all possible continuations and their frequencies.
        possible_tokens = self.distribution[tuple(tokens)]

        # If there are fewer continuations than the limit, return all of them.
        if len(possible_tokens) <= self.limit:
            return list(possible_tokens.keys())

        # Otherwise, we need to extract the 'self.limit' most frequent continuations.
        # 'heapq.nlargest' helps efficiently find the largest elements in a collection.
        # We use a lambda function to specify that we're comparing the values (frequencies) in the dictionary.
        most_frequent_tokens = heapq.nlargest(self.limit, possible_tokens, key=lambda x: possible_tokens[x])

        return most_frequent_tokens
    def calculate_self_entropy(self):
        """
        Calculates the cross-entropy for the experiment set based on the trigram model.
        :return: The cross-entropy value.
        """
        n = len(self.experiment_set) - 2  # since we're working with trigrams
        if n <= 0:
            return 0  # Avoid division by zero or taking log of zero. Handle the edge case.

        total_log_probability = 0.0

        # Iterate through the experiment set with a step of 3 since it's a trigram model
        for i in range(n):
            # Extract trigram
            a1, a2, a3 = self.experiment_set[i], self.experiment_set[i+1], self.experiment_set[i+2]
            
            # Fetch the conditional probability of the trigram from the distribution
            bigram_prob = self.distribution.get((a1, a2), {})
            trigram_prob = bigram_prob.get(a3, 0)
            
            # To get conditional probability, we need to normalize by the sum of all possibilities for the bigram
            conditional_prob = trigram_prob / (sum(bigram_prob.values()) or 1)
            
            # The provided formula uses log base e. If the probability is 0, it's undefined, so we skip it.
            if conditional_prob > 0:
                total_log_probability += math.log(conditional_prob)

        # Compute the cross-entropy
        entropy = - total_log_probability / n
        return entropy
    def calculate_cross_entropy(self,data):
        """
        Predicts the next set of tokens based on the input.
        :param tokens: List of tokens to base the prediction on.
        :return: A list of predicted tokens.
        """
        # Determine the length for 10% of the data
        ten_percent_length = len(data) // 10
        if ten_percent_length <= 2:
            return 0  # Avoid taking log of zero and ensure we have at least one trigram. Handle the edge case.

        # Randomly select a starting point for the 10% segment
        start_idx = np.random.randint(0, len(data) - ten_percent_length + 1)
        subset = data[start_idx:start_idx + ten_percent_length]

        n = len(subset) - 2  # since we're working with trigrams
        total_log_probability = 0.0

        # Iterate through the subset with a step of 1 since it's a trigram model
        for i in range(n):
            # Extract trigram
            a1, a2, a3 = subset[i], subset[i + 1], subset[i + 2]

            # Fetch the conditional probability of the trigram from the distribution
            bigram_prob = self.distribution.get((a1, a2), {})
            trigram_prob = bigram_prob.get(a3, 0)

            # To get conditional probability, we need to normalize by the sum of all possibilities for the bigram
            conditional_prob = trigram_prob / (sum(bigram_prob.values()) or 1)

            # The provided formula uses log base e. If the probability is 0, it's undefined, so we skip it.
            if conditional_prob > 0:
                total_log_probability += math.log(conditional_prob)

        # Calculate the cross-entropy according to the formula
        cross_entropy = -total_log_probability / n
        return cross_entropy

    def calculate_avg_accuracy(self):
        """
        Calculate the average prediction accuracy based on the experiment_set.
        :return: The average accuracy as a float.
        """
        
        correct_predictions = 0
        total_predictions = 0
        
        # Iterate over the experiment_set with an index
        for i in range(len(self.experiment_set) - 2):
            # Get the actual tokens
            actual_tokens = self.experiment_set[i:i+3]
            
            # Get the predicted tokens using the first two tokens
            predicted_tokens = self.predict(tuple(actual_tokens[:2]))
            
            # Check if the third actual token is in the predicted tokens
            if actual_tokens[2] in predicted_tokens:
                correct_predictions += 1
            
            total_predictions += 1
        
        # Return the average accuracy
        return correct_predictions / total_predictions if total_predictions > 0 else 0
