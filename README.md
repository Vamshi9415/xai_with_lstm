The architecture in this notebook is a two-layer LSTM neural network with an attention mechanism, designed for text classification (spam detection). Here’s a simple breakdown:

Embedding Layer: Converts each word (token) in the SMS message into a dense vector, allowing the model to learn word relationships.
SpatialDropout1D: Randomly drops entire 1D feature maps, helping prevent overfitting.
First LSTM Layer: Processes the sequence of word vectors, capturing patterns and dependencies in the text. It returns sequences so the next LSTM can process the full output.
Second LSTM Layer: Further processes the sequence, but with fewer units, and also returns sequences for the attention layer.
Attention Layer: Calculates attention weights for each time step (word), allowing the model to focus on the most important parts of the message for classification.
Dense Layers: After attention, the output is passed through a dense (fully connected) layer with ReLU activation, followed by dropout for regularization.
Output Layer: A single neuron with sigmoid activation outputs the probability that the message is spam.
This setup helps the model not only learn the sequence of words but also focus on the most relevant words for making its decision


After the model makes predictions, I use LIME to explain why it made certain decisions, especially for messages it got wrong. For each misclassified message, I look at which words or tokens had the biggest impact on the model’s choice. I sort these features by how important they were, show them in a simple bar chart, and print out the top ones. This helps me see if the model is focusing on the right parts of the message or getting confused by certain words. By doing this for a few examples, I can better understand the model’s mistakes and think about how to improve it next time.
