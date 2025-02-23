#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import pandas as pd

if __name__ == '__main__':
    with open('data/GazeT/dataset/metadata.json', 'r') as f:
        metadata_json = json.load(f)
    metadata_df = pd.DataFrame(metadata_json)
    print(metadata_df.head())