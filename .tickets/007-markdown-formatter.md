# Реализация Markdown Formatter

- Номер: 007
- Depends on: [006]
- Спецификация: `.specification/requirements.md#FR-012, FR-013` и `.specification/architecture.md#Markdown Formatter`
- Тип: backend
- Приоритет: P1
- Статус: todo

## Definition of Ready
- Все зависимости закрыты; ссылки на спецификацию валидны; Inputs/Outputs описаны; команды запуска/проверки даны.

## Контекст
- Необходимо реализовать компонент для форматирования ответов в Markdown и сохранения в файлы
- Документы: `.specification/requirements.md` - функциональные требования FR-012, FR-013 (строки 77-91)
- Документы: `.specification/architecture.md` - раздел "Markdown Formatter" (строки 526-615)

## Цель
Создать модуль для форматирования ответов RAG системы в Markdown с поддержкой сохранения в файлы.

## Inputs
- Спецификация проекта: `.specification/requirements.md` - FR-012, FR-013
- RAG Engine: созданный в задаче 006
- Система конфигурации: созданная в задаче 002

## Outputs
- Модуль `markdown_formatter.py` с классом MarkdownFormatter
- Функции форматирования ответов в Markdown
- Функции сохранения ответов в файлы
- Автоматическая генерация имен файлов
- Поддержка структурированного вывода

## Объем работ (Scope)
1) Создать класс MarkdownFormatter с настройкой директории вывода
2) Реализовать метод format_response для форматирования ответов в Markdown
3) Реализовать метод save_to_file для сохранения в файлы
4) Добавить автоматическую генерацию имен файлов на основе запроса
5) Реализовать метод format_sources_list для форматирования источников
6) Добавить поддержку заголовков, списков и цитат
7) Интегрировать с системой конфигурации
- Out of scope: CLI интерфейс (это отдельная задача)

## Acceptance criteria
- Команда: `python -c "from markdown_formatter import MarkdownFormatter; m = MarkdownFormatter(); result = {'answer': 'test', 'sources': []}; print('#' in m.format_response('test', result))"` → выводит True
- Команда: `python -c "from markdown_formatter import MarkdownFormatter; m = MarkdownFormatter(); result = {'answer': 'test', 'sources': []}; filepath = m.save_to_file('test query', result); print(filepath.endswith('.md'))"` → выводит True
- Файлы сохраняются в директории output/ с корректными именами

## Тесты
- Unit: тесты форматирования Markdown
- Integration: тесты сохранения в файлы
- E2E: тесты полного цикла форматирования и сохранения

## Команды для проверки
```bash
# проверка форматирования
python -c "from markdown_formatter import MarkdownFormatter; m = MarkdownFormatter(); result = {'answer': 'test', 'sources': []}; print('#' in m.format_response('test', result))"
# проверка сохранения
python -c "from markdown_formatter import MarkdownFormatter; m = MarkdownFormatter(); result = {'answer': 'test', 'sources': []}; filepath = m.save_to_file('test query', result); print(filepath.endswith('.md'))"
# проверка структуры файла
cat output/test_query_*.md
```

## Definition of Done
- Lint/type-check/tests/build OK
- Markdown Formatter реализован согласно спецификации
- Функциональные требования FR-012, FR-013 выполнены
- Конфигурация интегрирована
- Статус задачи обновлён на `done`

## Риски и допущения
- Предполагается наличие библиотеки markdown==3.5.1
- Предполагается наличие директории output/ для сохранения файлов
- Предполагается корректная кодировка UTF-8 для файлов

## Ссылка на следующую задачу
- 008

