# Naive Bayes Classifier

Text classification for the 20 Newsgroups Kaggle Dataset. Created for a Intro to Artificial Intelligence class at Bard College.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. The assumption here is you are using a unix environment with the latest version of python3.

### Prerequisites

Python 3.4 is by default shipped with Ubuntu. If you have macOS, you may have to set up the Homebrew package manager. Simply open terminal and run

```
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ export PATH=/usr/local/bin:/usr/local/sbin:$PATH
$ brew install python3
```
This may take a minute or two

### Installing

Installation process is simply cloning this repository. Below is a list of the files in the repository, as well as a short explanation describing the purpose of the file.

## Running the tests

The way data text data is stored corresponding to their category is as follows: the first word of each line represents what newsgroup the string of words belong to - and the the string of words following behind it are the words in the newsgroup.
```
alt.atheism	alt atheism faq atheist resourc archiv name atheism resourc alt atheism archiv name  ... 
```

### Test Results

Results of running the test file on the classifier trained using the training dataset should result in the following results for each category...

### Next steps

Implementation of Inverse Document Frequency to weigh the probabilities corresponding to how frequently the words appear in all the categories

## Built With

* [20 Newsgroups | Kaggle](https://www.kaggle.com/crawford/20-newsgroups) - Newsgroups text and training dataset

## Authors

* **Zisheng Jason Chang** - [jzisheng](https://github.com/jzisheng)

## Acknowledgments

* Much thanks to Sven Anderson for lecturing this course, and cleaning up the dataset and formatting it for usage in the course.
