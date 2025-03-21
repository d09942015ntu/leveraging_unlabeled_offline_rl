# [Implementation] Leveraging Unlabeled Data Sharing through Kernel Function Approximation in Offline Reinforcement Learning

[https://arxiv.org/abs/2408.12307](https://arxiv.org/abs/2408.12307)

## Set up Virtual Environment

```sh
python3 -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

## Run the Experiment of Asymptotic Behavior

```sh
python3 experiment_asym.py #Run Experiment
python3 latex_asym.py      #Draw latex plot
```

## Run the Experiment of Comparison between Finite Dimensional and Kernel Features

```sh
bash run_carpole.sh        #Run Experiment
python3 latex_asym.py      #Draw latex plot
```
