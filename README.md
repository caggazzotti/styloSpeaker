# StyloSpeaker
### Stylometric speaker attribution from speech transcripts.

This is the official repo for the StyloSpeaker paper (link coming soon). This code is setup to extract stylometric features from the Fisher speech transcripts corpus, specifically the speaker verification trials from the paper [Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?](https://arxiv.org/abs/2311.07564) and its associated [GitHub](https://github.com/caggazzotti/speech-attribution), but can be adapted to other speech transcript datasets.  

## Data 

To obtain the Fisher speaker verification trials used in the paper, follow the directions in the [attribution of transcribed speech GitHub](https://github.com/caggazzotti/speech-attribution).  

**NOTE**: The Fisher data require an LDC subscription to [Fisher English Training Speech Part 1 Transcripts](https://catalog.ldc.upenn.edu/LDC2004T19) and [Fisher English Training Part 2 Transcripts](https://catalog.ldc.upenn.edu/LDC2005T19).

To use your own speech transcript data, put the data in the following verification trial format, where label `0` is for negative (different speaker) trials and `1` is for positive (same speaker) trials. The value for each "call" key is a list of utterances for a particular speaker in a call. Do this for both the training trials and the test trials.

```
 trials = [{"label": 0,
            "call 1": ["utterance 1", "utterance 2",...],
            "call 2": ["utterance 1", "utterance 2",...]},
            {"label": 1,
            "call 1": ["utterance 1", "utterance 2",...],
            "call 2": ["utterance 1", "utterance 2",...]},
            ...]
```

## Installation

To create an environment with the required packages, run the following commands within the styloSpeaker directory:

```
conda create --name stylospkr python=3.9
conda install pip
pip install -r requirements.txt
```

**NOTE**: Not all of these packages are needed to run the basic code but were in the conda enviornment used for the experiments in the paper so were kept for continuity.

## Setup
The absolute paths for the overarching working directory for this project, the output directory (a subfolder of the overarching working directory), and the verification trials data (in the format shown above) need to be manually added by modifying the following path variables in `config.yaml`:

- `work_dir`: overarching project directory (`./styloSpeaker`)
- `output_dir`: subfolder of work_dir for outputting the results (`./styloSpeaker/output`)
- `trials_dir`: directory containing the verification trials data (`./speech-attribution/trials_data` if using Fisher data)

Adjust the other variables in the `config.yaml` file as needed:

- `encodings`: `bbn` = text-like; `ldc` = lowercase, limited punctuation (specifically for Fisher data)
- `levels`: `base`, `hard`, `harder` (difficulty levels based on amount of topic control; specifically for Fisher data)
- `feat_combos`: which method to use for combining features between two sides of a trial; `diffabs` = absolute difference between features (worked best for Fisher)
- `char_ngram_range`: character n-gram range inclusive
- `tok_ngram_range`: token n-gram range inclusive
- `pos_ngram_range`: POS tag n-gram range inclusive
- `which_clf`: which classifier to use; set up to use logistic regression with feature importance but can be replaced with a different binary classifier

The features should be extracted using a GPU if available. 

## Usage

**Step 1**: To run feature extraction for the verification trials and evaluate the classifier's performance, run the following:

```
  python stylometric_analysis.py config.yaml
```

This will produce the following files (for each encoding and level):

- feature pipeline (`.pkl`)
- X_train, X_test (`.npz`)
- y_train, y_test (`.pkl`)
- classifier fitted on the training data (`.joblib`)
- results file (`.txt`)
- feature importances (`.csv`)

**Step 2**: To produce heatmaps and negative vs. positive plots of the top features, run the following:

```
  python analyze_feature_impt.py config.yaml
```

## Citation
If you use our benchmark in your work, please consider citing our paper: (coming soon)


