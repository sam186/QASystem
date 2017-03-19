# Programming Assignment 4
## How to train
Remember to change the parameters in `code/train.py`. Run your model by:
    
    $ python code/train.py

## How to check locally
1. python process_glove.py --glove_dir download
2. export CUDA_VISIBLE_DEVICES=''
   python code/qa_answer.py --train_dir train
3. python code/evaluate.py data/squad/dev-v1.1.json dev-prediction.json

## How to submit:
1. Change the parameters in `code/qa_answer.py`, make sure they're the same as what you used in `code/train.py`. You need to specify `context_maxlen`, `question_maxlen` (Cannot be None).

2. Make sure your model is runnable by running

    $ python code/qa_answer.py

3. Run the submission script by the following command. You'll need to log in to codalab. This script will block until the job is complete.

    $ ./codalab_run-predict.sh

4. To submit sanity-check, run the following command. Visit [Codalab](https://worksheets.codalab.org/) to see results.

    $ cl edit run-predict -T cs224n-win17-submit-sanity-check

5. To submit dev 

    $ cl edit run-predict -T cs224n-win17-submit-dev

6. To submit test

    $ cl edit run-predict -T cs224n-win17-submit-test

