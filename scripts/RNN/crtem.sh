#!/bin/sh

python /people/mesa/Desktop/scripts/RNN/create_embeddings.py $SGE_TASK_ID tests
python /people/mesa/Desktop/scripts/RNN/create_embeddings.py $SGE_TASK_ID train
python /people/mesa/Desktop/scripts/RNN/create_embeddings.py $SGE_TASK_ID dev
