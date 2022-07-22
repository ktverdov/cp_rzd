**Данные:**

[train](https://lodmedia.hb.bizmrg.com/case_files/766370/train_dataset_train.zip "train"), [test](https://lodmedia.hb.bizmrg.com/case_files/766370/test_dataset_test.zip "test")

( ожидаемое положение / можно задать в конфигах ):

```shell
./data/train/images/
./data/train/mask/
./data/test
```

**Запуск:**

```python
python3 -m venv rzd_env
source rzd_env/bin/activate

pip install -r requirements.txt 
pip install -e .

./prepare.sh
./train_pl.sh
./inference_pl.sh
```

[Чекпоинты итоговых моделей (Release v-1.0_pl)](https://github.com/ktverdov/cp_rzd/releases/tag/v1.0-pl "Чекпоинты моделей")

Обучалось на: GPU: 2080ti ( CUDA 11.3 )
