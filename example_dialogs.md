```bash
CONSOLE % python repl.py --device cpu
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 30232.14it/s]
Neon Rift RAG-бот запущен.
Введите вопрос или /exit для выхода.
Параметры retrieval: top_k=3, min_relevance=0.50
```

```bash
Вы: Почему Lati слышит Serran Fylar?

Бот:
Краткое обоснование:
1. Фрагмент «Serran Fylar» описывает, что личность Serran Fylar была оцифрована программой Mindharvester и сохранена в виде мнемокопии.
2. Далее говорится, что спустя десятилетия мнемокопия попала на мнемокристалл Eternal Shard.
3. После ограбления Obsidian Atrium мнемокристалл оказался в голове Lati, из-за чего Lati слышит Serran Fylar.

Ответ:
Lati слышит Serran Fylar, потому что мнемокопия Serran Fylar находится в его голове на мнемокристалле Eternal Shard.

Источники:
- Serran Fylar; 02_serran_fylar.md; 2:0000

[retrieval]
- 0.5871 | Serran Fylar | 02_serran_fylar.md | chunk=2:0000
```

```bash
Вы: Какая организация поддерживает Obsidian Veil?

Бот:
Краткое обоснование:
1. Фрагмент «Obsidian Veil» описывает цифровой барьер.
2. В нём прямо сказано, что систему поддерживает LociGuard.

Ответ:
Цифровой барьер Obsidian Veil поддерживает организация LociGuard.

Источники:
- Obsidian Veil; 25_obsidian_veil.md; 25:0000

[retrieval]
- 0.6382 | Obsidian Veil | 25_obsidian_veil.md | chunk=25:0000
- 0.5107 | Obsidian Seer | 12_obsidian_seer.md | chunk=12:0000
```

```bash
Вы: Кто контролирует Junkhaven?

Бот:
Краткое обоснование:
1. В источнике «Junkhaven» прямо указано, что территорию контролирует вооружённая организация SHOGS.
2. Также в фрагменте упоминается, что фактическим правителем территории является бывший военный Serris Dane.

Ответ:
Junkhaven контролирует организация SHOGS, а фактическим правителем территории является Serris Dane.

Источники:
- Junkhaven; 22_junkhaven.md; 22:0000

[retrieval]
- 0.6655 | Junkhaven | 22_junkhaven.md | chunk=22:0000
```

```bash
Вы: Кто создал Mindharvester и для чего он используется?

Бот:
Краткое обоснование:
1. В фрагменте «Mindharvester» указано, что программу создала wireseer Kira Lokai.
2. Изначально программа предназначалась для копирования сознания в цифровую среду, но Obsidian Dynamics превратила её в средство извлечения личности.
3. В фрагменте «Kira Lokai» подтверждается, что Kira является создателем программы Mindharvester, и изначально она предназначалась для переноса сознания в цифровую среду.

Ответ:
Mindharvester создала wireseer Kira Lokai. Изначально программа предназначалась для переноса сознания в цифровую среду, но Obsidian Dynamics превратила её в средство извлечения личности.

Источники:
- Mindharvester; 24_mindharvester.md; 24:0000
- Kira Lokai; 09_kira_lokai.md; 09:0000

[retrieval]
- 0.7187 | Mindharvester | 24_mindharvester.md | chunk=24:0000
- 0.6354 | Kira Lokai | 09_kira_lokai.md | chunk=09:0000
```

```bash
Вы: Почему Paola Nias покинула Paulownians?

Бот:
Краткое обоснование:
1. В фрагменте «Paola Nias» указано, что Paola покинула клан после разногласий с лидером Deacon Voss.
2. Причина разногласий заключалась в том, что Deacon Voss искал стабильную работу у Ionis Bioworks, а Paola считала, что зависимость от корпорации угрожает свободе семьи.

Ответ:
Paola Nias покинула Paulownians из-за разногласий с лидером Deacon Voss по поводу сотрудничества с корпорацией Ionis Bioworks, так как считала, что зависимость от корпорации угрожает свободе семьи.

Источники:
- Paola Nias; 05_paola_nias.md; 5:0000

[retrieval]
- 0.6862 | Paola Nias | 05_paola_nias.md | chunk=5:0000
- 0.5701 | Paulownians | 19_paulownians.md | chunk=19:0000
```

```bash
Вы: Как называется столица планеты Ти'лора?

Бот:
Я не знаю. В базе знаний нет достаточно релевантной информации.

[retrieval] Подходящие чанки не найдены; YandexGPT не вызывался.
```

```bash
Вы: Какой гарантийный срок производитель предоставляет на Velochron?

Бот:
Краткое обоснование:
1. В источнике «Velochron» нет информации о гарантийном сроке на имплант.

Ответ:
Я не знаю. В базе знаний нет достаточно релевантной информации.

Источники:
- Velochron; 29_velochron.md; 29:0000

[retrieval]
- 0.6069 | Velochron | 29_velochron.md | chunk=29:0000
```

```bash
Вы: /exit
Бот: До свидания.
CONSOLE % 
```
