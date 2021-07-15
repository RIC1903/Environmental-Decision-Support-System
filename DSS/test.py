#!/usr/bin/env python
import logging

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES
GIT_DIR = "C:\Users\harih\Desktop\DSS\Env_DSS"
#NUM_TRAIN_RECORDS = 1000000  # Number of records to use in training the model.  I'd recommend 100-1000 times the number of diagnoses
NUM_TEST_RECORDS = 1000  # Number of records to use to use in testing the model
NUM_TRAIN_RECORDS = [1000, 2000, 4000, 6000, 8000, 10000, 13000, 16000, 25000, 32000, 50000]  # list of training record counts to test at
RESULTS_FILE = "C:\Users\harih\Desktop\DSS\Env_DSS\results.json"
LOG_LEVEL = logging.DEBUG


## IMPORTS


import imp
import operator
import json
import pandas as pd

## SETUP
__author__ = "Hariharan"
logging.basicConfig(level=LOG_LEVEL)

## EXECUTION




def main():
    logging.info('Beginning main loop.')

    # Generate Synthetic Data
    print "Loading the synthetic data generation module."
    fp, pathname, description = imp.find_module("synthetic", [GIT_DIR])
    synthetic = imp.load_module("synthetic", fp, pathname, description)
    # Create class instance
    print "Creating the synthetic data object 'data' and truth data."
    data = synthetic.test_data()

    print "Loading the model module."
    fp, pathname, description = imp.find_module("model", [GIT_DIR])
    model = imp.load_module("model", fp, pathname, description)

    data.records = list()
    results = dict()

    # Create records
    print "Starting the record creation and testing loop."
    for i in range(len(NUM_TRAIN_RECORDS)):
        if i == 0:
            record_increment = NUM_TRAIN_RECORDS[i]
        else:
            record_increment = NUM_TRAIN_RECORDS[i] - NUM_TRAIN_RECORDS[i - 1]
        print "Creating {0} more synthetic noisy records for a total of {1}.".format(record_increment, len(data.records) + record_increment)
        data.records = data.records + data.create_diagnosis_data(data.truth, record_increment, data.default)
        # Train a model based on the synthetic data
        # Create decision support system object
        print "Creating the medical decision support system object 'mdss'."
        mdss = model.decision_support_system()
        print "Creating the model."
        mdss.model = mdss.train_nx_model(data.records)

        # Use the model to make diagnoses and see how well it did
        print "Generating {0} testing records 'test_records' for {1} training records.".format(NUM_TEST_RECORDS, len(data.records))
        test_records = data.create_diagnosis_data(data.truth, NUM_TEST_RECORDS, data.default)

        print "Diagnosing the test records."
        truth_diagnoses = dict()
        predicted_diagnoses = dict()
        for j in range(len(test_records)):
            truth_diagnoses[j] = test_records[j].pop('diagnosis')
            predicted_diagnoses[j] = mdss.query_nx_model(test_records[j])

        print "Scoring the predictions"
        # Count number in top 1 and 5
        results[NUM_TRAIN_RECORDS[i]] = {
            'top': 0,
            'top5': 0,
            'locs': [],
            'scores': []
        }
        for j in range(len(predicted_diagnoses)):
#            predictions = sorted(predicted_diagnoses[i].items(), key=operator.itemgetter(1), reverse=True)
            predictions = pd.DataFrame(data={"diagnosis":predicted_diagnoses[j].keys(), "score":predicted_diagnoses[j].values()})
            predictions.sort(columns='score', ascending=False, inplace=True)
            predictions.reset_index(drop=True, inplace=True)
            if predictions is None:
                predictions[0][0] = [0,0]
#            if predictions[0][0] == truth_diagnoses[j]:
            if predictions.iloc[0,0] == truth_diagnoses[j]:
                results[NUM_TRAIN_RECORDS[i]]['top'] += 1
                results[NUM_TRAIN_RECORDS[i]]['top5'] += 1
#            elif truth_diagnoses[j] in [key for key, value in predictions[0:5]]:
            elif truth_diagnoses[j] in list(predictions.iloc[0:5,0]):
                results[NUM_TRAIN_RECORDS[i]]['top5'] += 1
            try:
#                loc = predictions.index(truth_diagnoses[j])
                loc = predictions[predictions.diagnosis == truth_diagnoses[j]].index.tolist()
                if len(loc) > 0:
                    loc = loc[0]
                    loc = int(loc) + 1  # because values starting at 0 will confuse people
                else:
                    loc = -1
            except ValueError:
                loc = -1
            results[NUM_TRAIN_RECORDS[i]]['locs'].append(loc)
            # find the score difference
            if loc == -1:
                score_diff = 1
            else:
                score_diff = round((predictions.iloc[0,1]-predictions.iloc[loc-1,1])/float(predictions.iloc[0,1]), 7)
            results[NUM_TRAIN_RECORDS[i]]['scores'].append(score_diff)

        print "Writing {0} training record results.".format(len(data.records))
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f)

    print "Score the predictions."
    pass # TODO: Notionally going to do this on number of accuracy of top score & accuracy of top 5 scores

    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()
