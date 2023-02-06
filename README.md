This project replicates the results of Mikolov et al, 2013. Efficient Estimation of Word Representations in Vector Space
Instructions on running:
Open up __main__.py and run using Python 3.8.7 (newer versions should work as well).
    - If you want to designate a custom text corpus, change the 'data/bible.txt' to your desired corpus file. Note that the file must be in plaintext. Next, modify the line models.missingWordPredictor(3000, 10) and change the first parameter (3000) to the number of unique words that appear in your corpus

    - If you want to change the embedding dimension, change the second parameter in models.missingWordPredictor(3000, 10)
Run __main__.py. You should see a file named "trained_embedding_weights.pt" appear in the same directory as __main__.py after training finishes. To load this file, you must create an nn.Embedding objects with the same parameters as the one used in training.