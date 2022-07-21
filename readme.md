Данные:

```shell
./data/train/images/
./data/train/mask/
./data/test
```

Запуск:

```python
python3 -m venv rzd_env
source rzd_env/bin/activate

pip install -r requirements.txt 
pip install -e .

./prepare.sh
./train_pl.sh
./inference_pl.sh
```
