# Transfer-Learning
keras implementation of simple adversarial multi-task learning model

BiLSTM-MAX is used to encode sentence

gradient reversal layer is used to address minmax optimization problem

Usage:
python main.py --source_language=en --target_language=es

main.py: multi-task learning (NLI task)
singletask.py: single nli task
