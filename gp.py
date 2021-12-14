import torch
import numpy as np

import sys

sys.path.append("/home/alex/geo_prior/geo_prior_inat")
from geo_prior import models
from geo_prior import datasets as dt
from geo_prior import grid_predictor as grid
from geo_prior import utils


class GeoPriorModel:
    def __init__(self, path):
        print("Loading geospatial model: " + path)
        net_params = torch.load(path, map_location="cpu")
        self.params = net_params["params"]
        # only cpu plz
        self.params["device"] = "cpu"
        model_name = models.select_model(self.params["model"])
        self.model = model_name(
            num_inputs=self.params["num_feats"],
            num_classes=self.params["num_classes"],
            num_filts=self.params["num_filts"],
            num_users=self.params["num_users"],
            num_context=self.params["num_context"],
        ).to(self.params["device"])
        self.model.load_state_dict(net_params["state_dict"])
        self.model.eval()

    def predict(self, latitude, longitude):
        obs_loc = np.array([longitude, latitude])[np.newaxis, ...]
        obs_date = np.ones(1) * 0.5
        obs_loc, obs_date = utils.convert_loc_and_date(
            obs_loc, obs_date, self.params["device"]
        )
        loc_date_feats = utils.generate_feats(obs_loc, obs_date, self.params, None)

        with torch.no_grad():
            geo_pred = self.model(loc_date_feats)[0, :]
        geo_pred = geo_pred.cpu().numpy()
        geo_pred_dict = dict(zip(self.params["class_to_taxa"], geo_pred))

        # NOTE due to a problem during export, 3 taxa that are in the vision model
        # are not in this model. solve this by adding a static 1.0 prior for each of these tax
        # the taxon_ids in question are [428017, 358226, 61341]
        for missing_taxon_id in [428017, 358226, 61341]:
            print(
                "WARNING *** ADDING synthetic 1.0 score for taxon id {}".format(
                    missing_taxon_id
                )
            )
            geo_pred_dict[missing_taxon_id] = 1.0

        return geo_pred_dict
