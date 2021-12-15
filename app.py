import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
import requests
import redis

import base64
import io
import sys
import os
import hashlib
import yaml
import uuid
import magic

from taxonomy import Taxonomy
from gp import GeoPriorModel
from image_utils import path_for_photo_id

config = yaml.safe_load(open("config.yml"))


# clean up the static directory on launch
for f in os.listdir(config["cv_upload_tmp_path"]):
    if f != ".gitkeep":
        os.remove(os.path.join(config["cv_upload_tmp_path"], f))

# don't touch the GPU, if there is one
tf.config.set_visible_devices([], "GPU")

# load models & related taxonomy
inat_cv_model = tf.keras.applications.Xception(
    classes=int(config["cv_num_classes"]), weights=config["cv_model_path"]
)
geo_prior_model = GeoPriorModel(config["geo_model_path"])
taxonomy = Taxonomy(config["taxonomy_csv_file"])

# redis for taxon photos
taxon_photo_redis = redis.Redis(
    host=config["taxon_photos_redis_host"],
    port=config["taxon_photos_redis_port"],
    db=config["taxon_photos_redis_db"],
)


def combine_scores(vision_pred_dict, geo_pred_dict, method="multiply"):
    """
    combine vision and geo scores. combines two dicts containing
    { taxon_id: score }
    expects to find matching taxon ids in each.
    """

    if method != "multiply":
        assert (False, "only multiply has been implemented")

    combined_scores_dict = {}
    for taxon_id in vision_pred_dict:
        vision_score = vision_pred_dict[taxon_id]
        geo_score = geo_pred_dict[taxon_id]
        combined_scores_dict[taxon_id] = vision_score * geo_score

    # normalize the scores
    combined_sum = sum(combined_scores_dict.values())
    norm_combined_scores_dict = {}
    for taxon_id in combined_scores_dict:
        combined_score = combined_scores_dict[taxon_id]
        combined_score_norm = combined_score / combined_sum
        norm_combined_scores_dict[taxon_id] = combined_score_norm

    return (combined_scores_dict, norm_combined_scores_dict)


def photo_urls_for_taxon_id(taxon_id):
    """
    fetch taxon photo urls and cache them in redis
    """
    tp_key = "{}-taxon-photo".format(taxon_id)

    if not taxon_photo_redis.exists(tp_key):
        url = "https://api.inaturalist.org/v1/taxa/{}".format(taxon_id)
        r = requests.get(url)
        for taxon in r.json()["results"]:
            tid = taxon["id"]
            tp_urls = [tp["photo"]["medium_url"] for tp in taxon["taxon_photos"]]
            tp_key = "{}-taxon-photo".format(tid)
            taxon_photo_redis.set(tp_key, tp_urls[0])

    return taxon_photo_redis.get(tp_redis_key).decode()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("iNat CV & Geo Prior Dashboard"),
        html.Br(),
        html.P(
            "Enter an observation id and click the Classify button to see CV & geospatial suggestions via the geo prior model."
        ),
        html.P(
            "This may take a moment as we have to download the photo from the iNat api and process the results."
        ),
        html.Br(),
        html.Div(
            [
                "Observation Id",
                dcc.Input(id="input-obs-id", value="", type="text"),
                html.Button("Classify", id="btn-obs-id-classify", n_clicks=0),
            ]
        ),
        html.Br(),
        html.Div(id="output-obs-id"),
    ]
)


def classify_report_img_file(img_file_path, med_url, latitude, longitude):
    img = tf.io.read_file(img_file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.central_crop(img, 0.875)
    img = tf.image.resize(img, [299, 299], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.expand_dims(img, 0)

    # make vision & geo predictions
    vision_preds = inat_cv_model.predict(img)[0]
    vision_scores_dict = dict(zip(taxonomy.cv_class_to_tax, vision_preds))
    agg_vision_scores = taxonomy.aggregate_scores(vision_preds, taxonomy.life)
    best_vision_branch = taxonomy.best_branch_from_scores(agg_vision_scores)

    geo_pred_dict = geo_prior_model.predict(longitude, latitude)

    # combine & normalize the vision and geo scores
    combined_scores_dict, norm_combined_scores_dict = combine_scores(
        vision_scores_dict, geo_pred_dict
    )
    agg_comb_norm_scores = taxonomy.aggregate_scores(
        list(norm_combined_scores_dict.values()), taxonomy.life
    )
    best_comb_norm_branch = taxonomy.best_branch_from_scores(agg_comb_norm_scores)

    norm_sorted_combined = {
        k: v
        for k, v in sorted(
            norm_combined_scores_dict.items(), key=lambda item: item[1], reverse=True
        )
    }
    most_likely_combined = list(norm_sorted_combined.keys())[:5]

    taxon_photos = [photo_urls_for_taxon_id(tid) for tid in most_likely_combined]

    vision_scores = [vision_scores_dict[tid] for tid in most_likely_combined]
    geo_scores = [geo_pred_dict[tid] for tid in most_likely_combined]
    combined_scores = [combined_scores_dict[tid] for tid in most_likely_combined]
    vision_taxon_names = [
        taxonomy.leaf_tax[taxonomy.leaf_tax.taxon_id == tid].iloc[0]["name"]
        for tid in most_likely_combined
    ]
    norm_scores = [norm_combined_scores_dict[tid] for tid in most_likely_combined]

    tbl_header = html.Thead(
        html.Tr(
            [
                html.Th("Taxon Id"),
                html.Th("Taxon Name"),
                html.Th("CV Score"),
                html.Th("Geo Prior"),
                html.Th("Combined Score"),
                html.Th("Normalized Combined Score"),
                html.Th("Taxon Photo"),
            ]
        ),
    )

    tbl_body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(most_likely_combined[i]),
                    html.Td(vision_taxon_names[i]),
                    html.Td("{:.3}".format(vision_scores[i])),
                    html.Td("{:.3}".format(geo_scores[i])),
                    html.Td("{:.3}".format(combined_scores[i])),
                    html.Td("{:.3}".format(norm_scores[i])),
                    html.Td(
                        [
                            html.Img(src=taxon_photos[i], style={"width": 100}),
                        ]
                    ),
                ]
            )
            for i in range(5)
        ]
    )

    bb_theader = html.Thead(
        html.Tr(
            [
                html.Th("Name"),
                html.Th("Score"),
            ]
        ),
    )

    vbb_tbody = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(bb_node.name),
                    html.Td(bb_score),
                ]
            )
            for (bb_node, bb_score) in best_vision_branch
        ]
    )

    cbb_tbody = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(bb_node.name),
                    html.Td(bb_score),
                ]
            )
            for (bb_node, bb_score) in best_comb_norm_branch
        ]
    )

    return html.Div(
        [
            html.Img(src=med_url, style={"width": 300}),
            html.Br(),
            dbc.Table(
                [tbl_header, tbl_body],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Vision Best Branch"),
                            dbc.Table(
                                [bb_theader, vbb_tbody],
                                bordered=True,
                                hover=True,
                                responsive=True,
                                striped=True,
                            ),
                        ],
                        style={"float": "left", "width": "45%"},
                    ),
                    html.Div(
                        [
                            html.H4("Combined Best Branch"),
                            dbc.Table(
                                [bb_theader, cbb_tbody],
                                bordered=True,
                                hover=True,
                                responsive=True,
                                striped=True,
                            ),
                        ],
                        style={"float": "right", "width": "45%"},
                    ),
                ],
                style={"width": "100%"},
            ),
        ],
        style={"width": "100%"},
    )


# input from the obs id tab
@app.callback(
    Output("output-obs-id", "children"),
    Input("btn-obs-id-classify", "n_clicks"),
    State("input-obs-id", "value"),
)
def classify_obs_id(n_clicks, obs_id):
    if len(obs_id) == 0:
        return "Please enter an observation id"
    try:
        obs_id_int = int(obs_id)
    except ValueError:
        return "Can't convert that to an integer observation id"

    r = requests.get(
        "https://api.inaturalist.org/v1/observations/{}".format(obs_id_int)
    )
    if r.status_code != 200:
        return "Got a bad status code {}, are you sure that's valid observation id?".format(
            r.status_code
        )

    # get the obs json
    obs = r.json()["results"][0]
    location = obs["location"]
    if len(location) == 0:
        return "No location, sorry this dashboard won't work"
    latitude, longitude = (float(x) for x in location.split(","))

    obs_taxon_name = ""
    if obs["taxon"]:
        obs_taxon_name = obs["taxon"]["name"]

    if len(obs["observation_photos"]) == 0:
        return "No photos for that obs"

    sq_url = obs["observation_photos"][0]["photo"]["url"]
    med_url = sq_url.replace("square", "medium")

    extension = os.path.splitext(med_url)[1]
    image_uuid = str(uuid.uuid4())
    file_path = os.path.join(config["cv_upload_tmp_path"], image_uuid) + extension

    r = requests.get(med_url)
    with open(file_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

    mime_type = magic.from_file(file_path, mime=True)
    # attempt to convert non jpegs
    if mime_type != "image/jpeg":
        im = Image.open(file_path)
        rgb_im = im.convert("RGB")
        file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + ".jpg"
        rgb_im.save(file_path)

    return classify_report_img_file(file_path, med_url, latitude, longitude)


if __name__ == "__main__":
    # running the server with debug=True causes TF problems
    app.run_server(debug=False)
