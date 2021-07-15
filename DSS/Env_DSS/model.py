#!/usr/bin/env python

import numpy as np




# USER VARIABLES
NEODB = "http://192.168.121.134:7474/db/data"

# SET RANDOM SEED 
np.random.seed(5052015)

## TRUTH DATA STATIC VARIABLES, based on physician recommendation
DIAGNOSES = 10000
SIGNS = 3000
SYMPTOMS = 150
SIGNS_PER_DIAG_MEAN = 5.5
SIGNS_PER_DIAG_SD = 0.5  # Increased from 0.25 to give a bit of variance consistent with physician suggestion
SYMPTOMS_PER_DIAG_MEAN = 7 
SYMPTOMS_PER_DIAG_SD = 0.5  # Increased from 0.5 to give a variance consistent with physician suggestion
PERCENT_CONTINUOUS_SIGNS = 0.05
PERCENT_CONTINUOUS_SYMPTOMS = 0.05
PREFERENTIALLY_ATTACH_SIGNS = True
PREFERENTIALLY_ATTACH_SYMPTOMS = False
#SYMPTOM_VS_SIGN_RELATIVE_WEIGHT = .7 # symptom_weight/sign_weight i.e. we trust signs 100% and symptoms 70%
SYMPTOM_VS_SIGN_RELATIVE_WEIGHT = .6/.85 # symptom_weight/sign_weight i.e. we trust signs 100% and symptoms 70%


from py2neo import Graph as py2neoGraph
import networkx as nx
import argparse
import logging
from flask import Flask
from flask_restful import reqparse, Resource, Api, abort
from collections import defaultdict, Counter
import copy
import scipy.stats
from scipy.optimize import curve_fit

## SETUP
__author__ = "Hariharan"
# Parse Arguments (should correspond to user variables)
parser = argparse.ArgumentParser(description='This script processes a graph.')
parser.add_argument('-d', '--debug',
                    help='Print lots of debugging statements',
                    action="store_const", dest="loglevel", const=logging.DEBUG,
                    default=logging.WARNING
                   )
parser.add_argument('-v', '--verbose',
                    help='Be verbose',
                    action="store_const", dest="loglevel", const=logging.INFO
                   )
parser.add_argument('--log', help='Location of log file', default=None)
# <add arguments here>
parser.add_argument('db', help='URL of the neo4j graph database', default=NEODB)
#args = parser.parse_args()
## Set up Logging
#if args.log is not None:
#    logging.basicConfig(filename=args.log, level=args.loglevel)
#else:
#    logging.basicConfig(level=args.loglevel)
# <add other setup here>
# Connect to database
#G = neo4j.GraphDatabaseService(NEODB)
#g = nx.DiGraph()
#NEODB = args.db


## EXECUTION
class decision_support_system():
    model = None
    records_graph = None
    signs = None
    symptoms = None
    diagnoses = None


    def __init__(self):
        pass


    def build_model_nx(self, g1=None):
        g2 = nx.DiGraph()

        # in case records_graph was not filled in, use the one from the object
        if g1 is None:
            g1 = self.records_graph

        # Store the graph to an internal variable.  This could be used for
        relationships = set(g1.edges())

        # this is a bit of a hack with a 4 relationship filter set arbitrarily
        remove_rels = set()
        for relationship in relationships:
            if len(g1.edge[relationship[0]][relationship[1]]) <= 4:  # 4 is arbitrary @ 1 million records, tru are usually > 20 or 30 & false are normally < 5
                remove_rels.add(relationship)
        relationships = relationships.difference(remove_rels)


        # for each edge pair
        for relationship in relationships:
            edges = copy.deepcopy(g1.edge[relationship[0]][relationship[1]])
            # build a distribution (histogram) of the values
            values = Counter([edges[x]['value'] for x in edges])
            x = np.asarray(values.keys())
            y = np.asarray(values.values())
            y_norm = y/float(sum(y))
            mu, std = scipy.stats.norm.fit([edges[n]['value'] for n in edges])


            # create the normal fit
            # if there is data beyond mu +- std, keep it
            if len([n for n in x if n < mu - std]) > 0 and len([n for n in x if n > mu + std]) > 0:
                normalize = scipy.stats.norm(loc=mu, scale=std).pdf(mu)  # normalize to the mean
                f = lambda x_data : scipy.stats.norm(loc=mu, scale=std).pdf(x_data) / float(normalize)
            # else, scrap it and create a halfnorm (or just use the cdf
            else:
                # Can't get a good plot out of halfnorm so using full norm & normalizing the to the max in the data
                normalize = scipy.stats.norm(loc=mu, scale=std).cdf(max(x))  # normalze to the maximum of the data
                f = lambda x_data: scipy.stats.norm(loc=mu, scale=std).cdf(x_data) / float(normalize)


            # store density function on edge
            g2.add_edge(relationship[0], relationship[1], value = f, relationship_count = len(edges))

            #f when called with a value for the sign/symptom will return a confidence score that should be normalized to 1

        # add a type to each node to make coloring/filtering them easier
        for node in g2.nodes():
            g2.node[node]['type'] = node.split("_",1)[0]

        # for each node, count the number of relationships it had from g1
        for node in g2.nodes():
            type = g2.node[node]['type']
            if type in ['sign', 'symptom']:
                g2.node[node]['relationship_count'] = g1.out_degree(node)
            elif type in ['diagnosis']:
                g2.node[node]['relationship_count'] = g2.in_degree(node)


        # boom.  model.
        return g2

    def injest_records_nx(self, records):
        """
        :param records: Takes a list of medical records (synthetic or otherwise) in the form {diagnosis:"", signs:{}, symptoms:{}}
        :return: a networkx graph representing the records
        """
        # Graph 1 for all relationships
        g1 = nx.MultiDiGraph()
        # Define dictionaries to store the signs, symptoms, and diagnoses and cardinality
        signs = defaultdict(int)
        symptoms = defaultdict(int)
        diagnoses = defaultdict(int)



        # for each record, connect signs/symtoms to their respective diagnoses with value
        for record in records:
            diagnoses['diagnosis'] += 1
            for sign in record['signs']:
                g1.add_edge(sign, record['diagnosis'], value=record['signs'][sign])
                signs[sign] += 1
                if type(record['signs'][sign]) not in [int, float, np.float64]:
                    raise TypeError('Sign/Symptom {0} must be some type of number rather than {1} for {2}'.format(
                        sign, type(record['signs'][sign]), record))
            for symptom in record['symptoms']:
                g1.add_edge(symptom, record['diagnosis'], value=record['symptoms'][symptom])
                symptoms[symptom] +=1

        # Actual can be found by querying the degree of self.record_graph
        self.signs = signs.keys()
        self.symptoms = symptoms.keys()
        self.diagnoses = diagnoses.keys()
        self.records_graph = g1
        return g1


    def query_nx_model(self, record):

        # get potential diagnoses
        potential_diagnoses = set()
        for sign in record['signs']:
            if self.model.has_node(sign):
                potential_diagnoses = potential_diagnoses.union(self.model.successors(sign))
        for symptom in record['symptoms']:
            if self.model.has_node(symptom):
                potential_diagnoses = potential_diagnoses.union(self.model.successors(symptom))

        # filter the diagnoses by primary symptom
        pass  # TODO

        # score the diagnoses
        scores = defaultdict(float)
        for diagnosis in potential_diagnoses:
            sign_edge_cnt = 0
            symptom_edge_cnt = 0
            sign_score = 0
            for sign in record['signs']:
                if self.model.has_edge(sign, diagnosis):
                    # edge values form signs/symptoms
                    sign_score += self.model.edge[sign][diagnosis]['value'](record['signs'][sign]) 
                    sign_score *= self.model.edge[sign][diagnosis]['relationship_count'] / \
                            float(self.model.node[diagnosis]['relationship_count'])
                    # weighted for relative importance of particular sign (i.e. chest pain > finger pain)
                    sign_score *= self.model.node[sign].get('relative_weight', 1)
                    # weighted less for symptoms
                    sign_score *= 1  # SYMPTOM_VS_SIGN_RELATIVE_WEIGHT value below is normalized to this one (1)
            symptom_score = 0
            for symptom in record['symptoms']:
                if self.model.has_edge(symptom, diagnosis):
                    symptom_score += self.model.edge[symptom][diagnosis]['value'](record['symptoms'][symptom])
                    try:
                        symptom_score *= self.model.edge[symptom][diagnosis]['relationship_count'] / \
                                float(self.model.node[diagnosis]['relationship_count'])
                    except:
                        print diagnosis, symptom
                        raise
                    symptom_score *= self.model.node[symptom].get('relative_weight', 1)
                    symptom_score *= SYMPTOM_VS_SIGN_RELATIVE_WEIGHT

            scores[diagnosis] = sign_score + symptom_score

        return scores


    def train_nx_model(self, records):
        # From the list of records, build the records graph
        self.records_graph = self.injest_records_nx(records)
        g1 = self.records_graph

        # From the records graph, build the graph model
        self.model = self.build_model_nx(g1)
        return self.model



def main():
    logging.info('Beginning main loop.')

    logging.info('Main loop complete.')
if __name__ == "__main__":
    main()    
