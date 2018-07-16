#!/bin/bash

cd ./data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf ./cifar-10-python.tar.gz
rm -f ./cifar-10-python.tar.gz
