import pandas as pd
import warnings
import random
import spacy
import matplotlib.patches as mpatches
from cffi.backend_ctypes import xrange
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
TEST_REVIEW = "ok"
precisionValues, recallValues, fScoreValues, lossValues = [], [], [], []


# Function to load the data
def load_data(data_directory: str = "SA/Musical_instruments_reviews.csv", split: float = 0.2) -> tuple:
    # Read data
    df = pd.read_csv(data_directory)

    # Check data for unique values and NA counts
    print("--------------------- DATASET ANALYSIS ------------------------")
    print(f"Number of unique customers in the dataset : {len(df['reviewerID'].unique())}")
    print(f"Number of unique products that were reviewed : {len(df['asin'].unique())}")
    print(f"\nColumns having blank values:")
    print(df.isna().sum())

    # Store Review text and Summary into a single column
    df['review'] = df['reviewText'] + ' ' + df['summary']

    # Get only the columns that are required and drop NA values
    df = df[['review', 'overall']].dropna()

    # Plot the ratings
    ax = df.overall.value_counts().plot(kind='bar')
    fig1 = ax.get_figure()
    fig1.savefig("RatingsScore.png")

    # Change the ratings Rating(>3) is 1 and Rating(<=3) is 0
    df.overall[df.overall <= 3] = 0
    df.overall[df.overall > 3] = 1

    # Plot the boolean ratings
    ax = df.overall.value_counts().plot(kind='bar')
    fig2 = ax.get_figure()
    fig2.savefig("score_boolean.png")

    # To balance classes, select equal samples from each class
    pos_df = df[df.overall == 1][:1200]
    neg_df = df[df.overall == 0][:1200]

    # Integrate into single TRAIN dataset
    train = pos_df.append(neg_df)

    # Get target and train dataset
    train_target = train['overall']

    # Split into train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(
        train, train_target, random_state=0, shuffle=True, test_size=split)

    # Create list of dictionary values (review text, category label) for spaCy to use later
    train_reviews = []
    for item in x_train.iterrows():
        text = item[1][0]
        rating = int(item[1][1])
        text = text.replace("<br />", "\n\n")
        if text.strip():
            spacy_label = {
                "cats": {
                    "pos": rating == 1,
                    "neg": rating == 0
                }
            }
        train_reviews.append((text, spacy_label))

    test_reviews = []
    for item in x_test.iterrows():
        text = item[1][0]
        rating = int(item[1][1])
        text = text.replace("<br />", "\n\n")
        if text.strip():
            spacy_label = {
                "cats": {
                    "pos": rating == 1,
                    "neg": rating == 0
                }
            }
        test_reviews.append((text, spacy_label))
    return train_reviews, test_reviews


# Function to train the model
def train_model(training_data: list, test_data: list, iterations: int = 20) -> None:
    # Build NLP pipeline : Use text category if it exists, otherwise create it
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat, exclude others
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]

    # This will disable all other pipes other than TextCat
    with nlp.disable_pipes(training_excluded_pipes):
        # The initial optimizer
        optimizer = nlp.begin_training()

        # Training loop
        print("Beginning training...")

        # A generator that yields infinite series of input numbers - to be used by minibatch() utility later
        batch_sizes = compounding(4.0, 32.0, 1.001)

        print("Iteration\tLoss\t\t\t\tPrecision\t\t\tRecall\t\t\t\tF-score")
        # Training loop
        for i in range(iterations):

            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)

            # Evaluate the model
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                # This is to plot these values on a graph
                lossValues.append(loss['textcat'])
                precisionValues.append(evaluation_results['precision'])
                recallValues.append(evaluation_results['recall'])
                fScoreValues.append(evaluation_results['f-score'])
                print(
                    f"{i}\t\t\t{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("NLP_Model_AmazonReviews")


# Function to evaluate the model
def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)  # iterate tokenized reviews without keeping all in memory
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                    predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


# Function to test model for an input sample review
def test_model(input_data: str = TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("NLP_Model_AmazonReviews")
    # Generate prediction

    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )


# Function to plot Losses, Precision, Accuracy and F-scores on a graph
def plot_results():
    plt.style.use('seaborn')
    # legends
    green_patch = mpatches.Patch(color="Green", label="Precision")
    blue_patch = mpatches.Patch(color="Blue", label="Recall")
    red_patch = mpatches.Patch(color="Red", label="F-Score")

    plot1 = plt.figure(3)
    plt.plot(precisionValues, color="Green")
    plt.plot(recallValues, color="Blue")
    plt.plot(fScoreValues, color="Red")
    plt.legend(handles=[green_patch, blue_patch, red_patch])
    plt.xlabel('Timeline')
    plt.ylabel('Accuracy Report values')

    plot2 = plt.figure(4)
    plt.plot(lossValues, color="Black")
    plt.xlabel('Timeline')
    plt.ylabel('Losses')
    plt.show()


train, test = load_data()
train_model(train, test, 25)
plot_results()
# review = input("\nPlease enter your feedback comments : ")
# test_model(review)