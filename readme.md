# Implementing "unlearnable" poisoning attack

## Requirements
```
pip install -r requirements.txt
```

## Usage
```
python main.py
```

## Configuration
- Min-min attack without model resetting: `python main.py`
- Min-min attack with model resetting: `python main.py --reset_model Y --atk_train_loops 10 --atk_adj_loops 3`
- Min-max attack without model resetting: `python main.py --attack_type min_max`
- Min-max attack with model resetting: `python main.py --attack_type min_max --reset_model Y --atk_train_loops 10 --atk_adj_loops 3`
- Min-min attack in series: `python main.py --train_parallel N`
- Min-max attack in series: `python main.py --attack_type min_max --train_parallel N`

