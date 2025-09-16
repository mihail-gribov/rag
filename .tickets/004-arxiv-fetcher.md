# Реализация arXiv Fetcher

- Номер: 004
- Depends on: [003]
- Спецификация: `.specification/requirements.md#FR-001, FR-002, FR-003` и `.specification/architecture.md#arXiv Fetcher`
- Тип: backend
- Приоритет: P1
- Статус: todo

## Definition of Ready
- Все зависимости закрыты; ссылки на спецификацию валидны; Inputs/Outputs описаны; команды запуска/проверки даны.

## Контекст
- Необходимо реализовать компонент для поиска и загрузки статей с arXiv.org
- Документы: `.specification/requirements.md` - функциональные требования FR-001, FR-002, FR-003 (строки 5-26)
- Документы: `.specification/architecture.md` - раздел "arXiv Fetcher" (строки 294-367)

## Цель
Создать модуль для поиска статей на arXiv.org и загрузки их PDF файлов с извлечением метаданных.

## Inputs
- Спецификация проекта: `.specification/requirements.md` - FR-001, FR-002, FR-003
- Система логирования: созданная в задаче 003
- Система конфигурации: созданная в задаче 002

## Outputs
- Модуль `arxiv_fetcher.py` с классом ArxivFetcher
- Функции поиска статей по запросу
- Функции загрузки PDF файлов
- Извлечение и сохранение метаданных статей
- Обработка ошибок загрузки

## Объем работ (Scope)
1) Создать класс ArxivFetcher с инициализацией директории для статей
2) Реализовать метод search_documents для поиска статей по запросу
3) Реализовать метод download_document для загрузки PDF файлов
4) Добавить извлечение метаданных статей (ID, название, авторы, аннотация, дата, категории)
5) Реализовать метод fetch_and_download для полного цикла загрузки
6) Добавить обработку ошибок и логирование
7) Интегрировать с системой конфигурации
- Out of scope: парсинг PDF файлов (это отдельная задача)

## Acceptance criteria
- Команда: `python -c "from arxiv_fetcher import ArxivFetcher; f = ArxivFetcher(); print(len(f.search_documents('machine learning', 5)))"` → выводит 5
- Команда: `python -c "from arxiv_fetcher import ArxivFetcher; f = ArxivFetcher(); docs = f.search_documents('RAG', 1); f.download_document(docs[0])"` → загружает PDF файл
- Логирование работает для всех операций

## Тесты
- Unit: тесты поиска статей
- Integration: тесты загрузки PDF файлов
- E2E: тесты полного цикла fetch_and_download

## Команды для проверки
```bash
# проверка поиска статей
python -c "from arxiv_fetcher import ArxivFetcher; f = ArxivFetcher(); print(len(f.search_documents('machine learning', 5)))"
# проверка загрузки
python -c "from arxiv_fetcher import ArxivFetcher; f = ArxivFetcher(); docs = f.search_documents('RAG', 1); f.download_document(docs[0])"
# проверка полного цикла
python -c "from arxiv_fetcher import ArxivFetcher; f = ArxivFetcher(); files = f.fetch_and_download('RAG', 3); print(len(files))"
```

## Definition of Done
- Lint/type-check/tests/build OK
- arXiv Fetcher реализован согласно спецификации
- Все функциональные требования FR-001, FR-002, FR-003 выполнены
- Логирование интегрировано
- Статус задачи обновлён на `done`

## Риски и допущения
- Предполагается доступ к arXiv.org API
- Предполагается наличие библиотеки arxiv==2.1.0
- Предполагается стабильная работа сети для загрузки PDF файлов

## Ссылка на следующую задачу
- 005

