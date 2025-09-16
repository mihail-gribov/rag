# Реализация Document Parser

- Номер: 005
- Depends on: [004]
- Спецификация: `.specification/requirements.md#FR-004` и `.specification/architecture.md#Document Parser`
- Тип: backend
- Приоритет: P1
- Статус: todo

## Definition of Ready
- Все зависимости закрыты; ссылки на спецификацию валидны; Inputs/Outputs описаны; команды запуска/проверки даны.

## Контекст
- Необходимо реализовать компонент для извлечения текста из PDF файлов и разбивки на чанки
- Документы: `.specification/requirements.md` - функциональное требование FR-004 (строки 27-32)
- Документы: `.specification/architecture.md` - раздел "Document Parser" (строки 369-411)

## Цель
Создать модуль для парсинга PDF файлов, извлечения текста и разбивки на чанки размером 512 токенов с перекрытием 50 токенов.

## Inputs
- Спецификация проекта: `.specification/requirements.md` - FR-004
- arXiv Fetcher: созданный в задаче 004
- Система логирования: созданная в задаче 003

## Outputs
- Модуль `document_parser.py` с классом DocumentParser
- Функции извлечения текста из PDF файлов
- Функции разбивки текста на чанки
- Обработка ошибок парсинга
- Извлечение метаданных документов

## Объем работ (Scope)
1) Создать класс DocumentParser с настройками размера чанков
2) Реализовать метод parse_pdf для извлечения текста из PDF
3) Реализовать метод parse_document для полного цикла парсинга
4) Добавить разбивку текста на чанки с перекрытием
5) Реализовать метод get_document_metadata для извлечения метаданных
6) Добавить обработку ошибок парсинга
7) Интегрировать с системой логирования
- Out of scope: векторизация чанков (это отдельная задача)

## Acceptance criteria
- Команда: `python -c "from document_parser import DocumentParser; p = DocumentParser(); chunks = p.parse_document('test.pdf'); print(len(chunks))"` → выводит количество чанков
- Команда: `python -c "from document_parser import DocumentParser; p = DocumentParser(); meta = p.get_document_metadata('test.pdf'); print(meta['file_type'])"` → выводит "pdf"
- Обработка ошибок работает для поврежденных PDF файлов

## Тесты
- Unit: тесты извлечения текста из PDF
- Integration: тесты разбивки на чанки
- E2E: тесты полного цикла парсинга

## Команды для проверки
```bash
# проверка парсинга PDF
python -c "from document_parser import DocumentParser; p = DocumentParser(); chunks = p.parse_document('test.pdf'); print(len(chunks))"
# проверка метаданных
python -c "from document_parser import DocumentParser; p = DocumentParser(); meta = p.get_document_metadata('test.pdf'); print(meta['file_type'])"
# проверка обработки ошибок
python -c "from document_parser import DocumentParser; p = DocumentParser(); p.parse_document('nonexistent.pdf')"
```

## Definition of Done
- Lint/type-check/tests/build OK
- Document Parser реализован согласно спецификации
- Функциональное требование FR-004 выполнено
- Логирование интегрировано
- Статус задачи обновлён на `done`

## Риски и допущения
- Предполагается наличие библиотеки PyPDF2==3.0.1
- Предполагается наличие библиотеки langchain==0.0.350 для разбивки текста
- Предполагается корректный формат PDF файлов

## Ссылка на следующую задачу
- 006

