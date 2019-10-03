import argparse
import logging
import joblib
import sys
import os
import logging
import pandas as pd
from kubernetes import client, config, watch

from kfmd import metadata
from datetime import datetime
import pandas
KUBEFLOW_METADATA_URL_PREFIX = "metadata-service.kubeflow:8080"
KUBEFLOW_METADATA_MY_NAMESPACE = "siva-moturi-pfe-gmail-com"

#training
import fairing
from fairing.builders.cluster import gcs_context
from fairing.builders.cluster.cluster import ClusterBuilder
from fairing.deployers.tfjob.tfjob import TfJob
from fairing.deployers.serving.serving import Serving
from fairing.preprocessors.function import FunctionPreProcessor
from fairing.preprocessors.function import BasePreProcessor

import subprocess



class Skywalker(object):
    
    def init_cloud(self):
        # Setting up google container repositories (GCR) for storing output containers
        # You can use any docker container registry istead of GCR
        self.GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
        self.DOCKER_REGISTRY = 'gcr.io/{}/fairing-job'.format(self.GCP_PROJECT)
        self.PY_VERSION = ".".join([str(x) for x in sys.version_info[0:3]])
        self.BASE_IMAGE = 'python:{}'.format(self.PY_VERSION)
        self.BASE_IMAGE_1 = "tensorflow/tensorflow:1.13.1-py3"
        self.MY_DEPLOYMENT_FILE_DICT = {os.path.abspath(os.path.dirname(__file__)) + "/core.py" : "/app/vslml/core.py",
                                        os.path.abspath(os.path.dirname(__file__)) + "/meta.py" : "/app/vslml/meta.py",
                                        os.path.abspath(os.path.dirname(__file__)) + "/io.py" : "/app/vslml/io.py",
                                       os.environ["GOOGLE_APPLICATION_CREDENTIALS"] : "/app/vslml/key.json"}

    def __init__(self, cred_key_json_path = "/home/jovyan/key.json"):
        #initialize the vessel service account based auth handshake & load kubectl config
        #cred_key_json_path="/home/jovyan/key.json" in notebooks OR /app/vslml/key.json on train/deploy container                           
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= cred_key_json_path
        config.load_kube_config()
        
        #deploy
        self.preprocessor=None
        self.builder=None
        self.pod_spec=None
        self.train_deployer=None
        self.infer_deployer=None
        self.endpoint=None
        
        #set logging level
        logging.basicConfig(format='%(message)s')
        logging.getLogger().setLevel(logging.INFO)
        
        self.init_cloud()

    def train_v1(self, mdl_trainer, train_data_file_list=['data/train.csv'], reqs_file='requirements.txt'):

        #TODO: optimize if sequence doesn't matter
        #logging.info("Creating docker context: %s", output_file)
        my_file_list = []
        my_file_list.append(mdl_trainer)
        my_file_list.extend(train_data_file_list)
        my_file_list.append(reqs_file)
        
        my_output_map = {key:value for (key,value) in self.MY_DEPLOYMENT_FILE_DICT.items()}
        #for f in train_data_file_list:
            #fname = f.split("/")[-1]
            #my_output_map[os.path.normpath(f)] = "/app/data/{0}".format(fname)
        
        self.preprocessor = BasePreProcessor(input_files=my_file_list,
                                             output_map = my_output_map)
        self.builder = ClusterBuilder(registry=self.DOCKER_REGISTRY, base_image=self.BASE_IMAGE, preprocessor=self.preprocessor,
                                pod_spec_mutators=[fairing.cloud.gcp.add_gcp_credentials_if_exists],
                                context_source=gcs_context.GCSContextSource())

        self.builder.build()
        self.pod_spec = self.builder.generate_pod_spec()
        self.train_deployer = TfJob(pod_spec_mutators=[fairing.cloud.gcp.add_gcp_credentials_if_exists],
                        worker_count=1, chief_count=0)
        j_name = self.train_deployer.deploy(self.pod_spec)
        return (self.train_deployer.job_id, j_name)
        
    def train(self, mdl_trainer, train_data_file_list=['train.csv'], reqs_file='requirements.txt'):

        my_file_list = []
        my_file_list.append(reqs_file)
        
        my_output_map = {key:value for (key,value) in self.MY_DEPLOYMENT_FILE_DICT.items()}
        for f in train_data_file_list:
            fname = f.split("/")[-1]
            my_output_map[os.path.normpath(f)] = "/app/data/{0}".format(fname)
        
        self.preprocessor = FunctionPreProcessor(function_obj=mdl_trainer,
                                                 output_map = my_output_map,
                                                 input_files= my_file_list)
        self.builder = ClusterBuilder(registry=self.DOCKER_REGISTRY, base_image=self.BASE_IMAGE, preprocessor=self.preprocessor,
                                pod_spec_mutators=[fairing.cloud.gcp.add_gcp_credentials_if_exists],
                                context_source=gcs_context.GCSContextSource())

        self.builder.build()
        self.pod_spec = self.builder.generate_pod_spec()
        self.train_deployer = TfJob(pod_spec_mutators=[fairing.cloud.gcp.add_gcp_credentials_if_exists],
                        worker_count=1, chief_count=0)
        j_name = self.train_deployer.deploy(self.pod_spec)
        return (self.train_deployer.job_id, j_name)
        
        
        
    def deploy(self, mdl_serve_class, mdl_serve, mdl_file_list=['model.dat'], reqs_file='requirements.txt'):
        
        my_file_list = []
        my_file_list.extend(mdl_file_list)
        my_file_list.append(reqs_file)
        
        my_output_map = {key:value for (key,value) in self.MY_DEPLOYMENT_FILE_DICT.items()}

        #Add vessel ML core files to deployment image (MY_DEPLOYMENT_FILE_DICT dictionary has source and desired dest paths)
        self.preprocessor = FunctionPreProcessor(function_obj=mdl_serve,
                                                 output_map = my_output_map,
                                                 input_files= my_file_list)
        self.builder = ClusterBuilder(registry=self.DOCKER_REGISTRY, base_image=self.BASE_IMAGE, preprocessor=self.preprocessor,
                                pod_spec_mutators=[fairing.cloud.gcp.add_gcp_credentials_if_exists],
                                context_source=gcs_context.GCSContextSource())

        self.builder.build()
        self.pod_spec = self.builder.generate_pod_spec()

        # TODO: Add support for deploying into custom user-provided namespaces (use namespace param of Serving class)
        self.infer_deployer = Serving(serving_class=mdl_serve_class)
        self.endpoint = self.infer_deployer.deploy(self.pod_spec)
        return self.endpoint
    
    def get_train_status(self, train_job_id):
        #kubectl -n siva-moturi-pfe-gmail-com get -o yaml tfjobs fairing-tfjob-hkvlx
        get_tf_job_status_cmd_template = "kubectl -n {0} get -o yaml tfjobs {1}"
        
        command_to_run = get_tf_job_status_cmd_template.format("siva-moturi-pfe-gmail-com", train_job_id).split(" ")
        cmd_out = subprocess.run(command_to_run, stdout=subprocess.PIPE)
        
        #TODO:parse command output and return meaninful response
        return cmd_out
        
    




