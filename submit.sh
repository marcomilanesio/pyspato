#!/bin/bash

zip models.zip models/*
spark-submit --master spark://obizo.inria.fr:7077 --num-executors 2 --py-files models.zip,utils.py main.py 2>/dev/null