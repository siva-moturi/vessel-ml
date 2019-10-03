import argparse
import logging
import sys
import os
import pandas as pd
from kubernetes import client, config, watch

from kfmd import metadata
from datetime import datetime
import pandas
KUBEFLOW_METADATA_URL_PREFIX = "metadata-service.kubeflow:8080"


#Vessel Kubeflow metdata accessor    
class R2d2(object):
    def __init__(self, workspace_name="vessel-xgboost-example", desc=""):
        self._ws_name=workspace_name
      
        self._ws= metadata.Workspace(
            # Connect to metadata-service in namesapce kubeflow in k8s cluster.
            backend_url_prefix=KUBEFLOW_METADATA_URL_PREFIX,
            name=self._ws_name)
        
        
    def get_metrics_data(self, model_id=None):
        if model_id is None:
            return pandas.DataFrame.from_dict(self._ws.list(metadata.Metrics.ARTIFACT_TYPE_NAME))
        else:
            df = pandas.DataFrame.from_dict(self._ws.list(metadata.Metrics.ARTIFACT_TYPE_NAME))
            return df[df["model_id"]== str(model_id)]
        
    def get_model_data(self, model_id=None):
        if model_id is None:
            return pandas.DataFrame.from_dict(self._ws.list(metadata.Model.ARTIFACT_TYPE_NAME))
        else:
            df = pandas.DataFrame.from_dict(self._ws.list(metadata.Model.ARTIFACT_TYPE_NAME))
            return df[df["model_id"]== str(model_id)]
        

        
        
class R2d2Logger(object):
    def __init__(self, workspace_name="vessel-xgboost-example",
                 owner="siva.moturi@pfizer.com",
                 execution_name_prefix="exec",
                 run_name_prefix ="run", desc=""):
        self.create_execution(workspace_name, owner, execution_name_prefix, run_name_prefix, desc)

    def create_execution(self, workspace_name, owner, execution_name_prefix, run_name_prefix, desc):
        
        self._ws_name=workspace_name
        self._owner = owner
        self._ws= metadata.Workspace(
            # Connect to metadata-service in namesapce kubeflow in k8s cluster.
            backend_url_prefix=KUBEFLOW_METADATA_URL_PREFIX,
            name=self._ws_name)
        
        self._r = metadata.Run(
            workspace=self._ws,
            name="run" + "-" + run_name_prefix  + "-" + datetime.utcnow().isoformat("T"),
            description="")

        self._exec = metadata.Execution(
            name = "execution" + "-" + execution_name_prefix + "-" + datetime.utcnow().isoformat("T"),
            workspace=self._ws,
            run=self._r,
            description="")
        
        self._model= None
        
    def log_model(self, name, framework_dict, hyper_param_dict, desc="", model_file_uri="", model_type=""):
        self._model = self._exec.log_output(metadata.Model(
            name=name,
            description=desc,
            owner=self._owner,
            uri=model_file_uri,
            model_type=model_type,
            training_framework=framework_dict,
            hyperparameters=hyper_param_dict,
            version=datetime.utcnow().isoformat("T")))
        
        return self._model
    
    def log_metrics(self, name, metrics_dict, desc="", uri="gcs://path/to/metrics"):
        metrics = self._exec.log_output(metadata.Metrics(
            name= name,
            owner=self._owner,
            description= desc,
            uri=uri,
            model_id=self._model.id,
            metrics_type=metadata.Metrics.VALIDATION,
            values = metrics_dict))
        return metrics

    
    